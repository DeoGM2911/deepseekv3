import torch
from torch import nn
from torch.nn import functional as F
import math
from .utils import rotary_emb


class ScaledDotProductAttention(nn.Module):
    r"""
    The scaled dot-product attention mechanism.
    
    Given a query matrix Q and a key matrix K, this module computes the attention scores
    as the dot product of Q and K, scaled by the square root of the dimensionality of the key vectors.
    The scores are then passed through a softmax function to obtain attention weights and optionally
    applies dropout for regularization. 

    Note that, this attention mechanism is **NOT** masked!
    
    \[
        \operatorname{softmax}\bigg(\dfrac{QK^T}{\sqrt{d_k}}\bigg)
    \]
    """
    def __init__(
        self,
        dropout=0.1
    ):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query,
        key,
        mask=None
    ):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)  # For numerical stability 

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        return attn_weights, scores


class MultiHeadLatentAttention(nn.Module):
    r"""
    Multi-head latent attention mechanism. The same as multi-head attention but introduce a latent representation
    for the keys and values. Save space by using latent representations across all heads during inference.

    For better understanding, one can review the core idea behind MLA for Deepseek V3. Below is a quick explanation.

    Given a input x, we will project it to a latent space z. Then, if we're running inference, we
    will extend the current key-value representation z to include the new one. Denote `B` the `batch_size`,
    `L` the input sequence length, `L_KV` the key-value sequence length, `H` the number of heads, `D` the 
    input dimension, and `D_KV` the latent dimension.

    x: (B, L, D) -> z: (B, L, D_KV) ---(concat along the L_KV dimension)---> z (B, L_KV + L, D_KV)
    This z will be stored in the decoder block during inference.

    Then, we will project z to the key-value space with W_k and W_v. x will be projected to the query space with W_q.
    We'll then perform multi-head attention. The output the the attention of the input witht the current key-value
    representation z.

    This attention scores will then be masked and multiply with the values. Note that, by doing this, we
    in essence perform multi-head self-attention on the new sequence (old sequence + x).

    Compared to the standard MHA, where the KV-cache is O(L * H * D_head) a.k.a the whole old attention map,
    in MLA, we only store the latent representation z, which is O(L * D_KV) ~ linear of the number of sequence.
    """
    def __init__(
        self,
        input_dim,
        latent_dim,
        num_heads,
        dropout=0.1
    ):
        super(MultiHeadLatentAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = latent_dim // num_heads
        
        # Latent projection
        self.latent_linear = nn.Linear(input_dim, latent_dim, bias=False)
        
        # Projection matrices
        self.W_q_up = nn.Linear(latent_dim, self.num_heads * self.hidden_dim, bias=False)
        self.W_q_down = nn.Linear(input_dim, latent_dim)
        self.q_proj_fused = False
        
        self.W_k_up = nn.Linear(latent_dim, self.num_heads * self.hidden_dim, bias=False)
        self.W_v_up = nn.Linear(latent_dim, self.num_heads * self.hidden_dim, bias=False)
        
        # RoPE projection matrices
        self.W_x_rope = nn.Linear(latent_dim, self.num_heads * self.hidden_dim, bias=False)
        self.W_k_rope = nn.Linear(input_dim, self.num_heads * self.hidden_dim, bias=False)

        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        # Output projection
        self.W_o = nn.Linear(num_heads * self.hidden_dim, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def _fuse_q_projection(self):
        """
        Fuse the q projection matrices after training.

        WARNING: ONLY CALL this after training during inference.
        """
        assert self.W_q_up.in_features == self.W_q_down.out_features
        self.W_q_eff = nn.Linear(self.W_q_down.in_features, self.W_q_up.out_features, bias=False)
        with torch.no_grad():
            self.W_q_eff.weight.copy_(self.W_q_up.weight @ self.W_q_down.weight)
        self.q_proj_fused = True

    def _seq_mask(
        self,
        scores,
        valid_lens: torch.Tensor
    ):
        """
        Perform masking
        """
        if valid_lens is None:
            return F.softmax(scores, dim=-1)
        # If valid_lens is of shape batch_size, repeat to match the number of heads
        else:
            shape = scores.shape

            # Enforce 2D valid_lens for causal and padding masking
            assert valid_lens.dim() == 2, "valid_lens must be of shape (batch_size, seq_len)"
            # repeat num_heads time for each batch
            valid_lens = valid_lens.repeat_interleave(self.num_heads, dim=0) # (batch_size * num_heads, num_queries)
            valid_lens = valid_lens.reshape(-1, 1)  # (batch_size * num_heads * num_queries, 1)
            scores = scores.reshape(-1, shape[-1])  # (batch_size * num_heads * num_queries, num_keys)
            masked_scores = scores.masked_fill(
                torch.arange(shape[-1], device=scores.device).unsqueeze(0) >= valid_lens,  # (batch_size * num_heads * num_queries, num_keys)
                -1e4
            )
            return F.softmax(masked_scores.reshape(shape), dim=-1)
    
    def forward(
        self,
        x,
        z,  # latent kv-cache
        key_rope,  # rope for keys, also cached
        valid_lens=None,
        inference=False
    ):
        # Latent representation for key, value
        if z is None:  # For training
            z = self.latent_linear(x)
        else:  # For KV-cache during inference
            z = torch.cat([z, self.latent_linear(x)], dim=1)  # Stack along sequence dimension
        
        # Reshape for projection
        batch_size, seq_len, kv_len = x.size(0), x.size(1), z.size(1)
        
        # Projection
        query_latent = self.W_q_down(x)
        if inference and self.q_proj_fused:
            query = self.W_q_eff(x)  # (batch_size, seq_len, num_heads * head_dim)
        else:
            query = self.W_q_up(query_latent)
        key = self.W_k_up(z)  # (batch_size, kv_len, num_heads * head_dim)
        value = self.W_v_up(z)  # (batch_size, kv_len, num_heads * head_dim)        
        
        x_rope = self.W_x_rope(query_latent).view(batch_size, seq_len, self.num_heads, self.hidden_dim)
        query_rope = rotary_emb(x_rope, torch.arange(seq_len, device=query_latent.device))
        
        # Projection for key RoPE
        k_rope = self.W_k_rope(x).view(batch_size, seq_len, self.num_heads, self.hidden_dim)
        k_rope = rotary_emb(k_rope, torch.arange(seq_len, device=k_rope.device))
        
        # Drop back to 3 dim
        query_rope = query_rope.view(batch_size, seq_len, self.num_heads * self.hidden_dim)
        k_rope = k_rope.view(batch_size, seq_len, self.num_heads * self.hidden_dim)

        # Cache the key rope
        if key_rope is None:
            key_rope = k_rope
        else:
            key_rope = torch.concat([key_rope, k_rope], dim=1)  # Add the new key rope to the existing key rope

        # Concat to get keys and queries along the head dimension
        key = torch.concat([key, key_rope], dim=-1)
        query = torch.concat([query, query_rope], dim=-1)
        
        # Reshape to separate heads: (batch_size, seq_len, num_heads, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        key = key.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, kv_len, self.num_heads, self.hidden_dim).transpose(1, 2)

        # Compute attention for all heads at once
        attn_weights, scores = self.attention(query, key)
        head_outputs = torch.matmul(self._seq_mask(scores, valid_lens), value)  # Shape: (batch_size, num_heads, seq_len, hidden_dim)
        
        # Reshape and concatenate head outputs
        head_outputs = head_outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, num_heads * hidden_dim)

        output = self.W_o(head_outputs)
        output = self.dropout(output)
        
        return output, z, key_rope, attn_weights, scores


# Unit test
if __name__ == "__main__":
    attention = MultiHeadLatentAttention(8, 16, 2)
    x = torch.ones((1, 3, 8))
    z, key_rope = None, None
    output, z, key_rope, attn_weights, scores = attention(x, z, key_rope)
    print("test 1")
    print(z.shape)
    print(key_rope.shape)
    print(output.shape)
    print(attn_weights.shape)
    print(scores.shape)

    # Continue stacking z
    print("test 2")
    output_2, z, key_rope, attn_weights_2, scores_2 = attention(x, z, key_rope) 
    print(z.shape)
    print(key_rope.shape)
    print(output_2.shape)
    print(attn_weights_2.shape)
    print(scores_2.shape)

    # Test valid lens - autoreg masked
    print("test 3")
    output_3, z, key_rope, attn_weights_3, scores_3 = attention(x, None, None, valid_lens=torch.arange(3, device=x.device).unsqueeze(0))
    print(z.shape)
    print(key_rope.shape)
    print(output_3.shape)
    print(attn_weights_3)
    print(scores_3)