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
        use_rotary_emb=True,
        dropout=0.1
    ):
        super(MultiHeadLatentAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = latent_dim // num_heads
        
        # Latent projection
        self.latent = nn.Linear(input_dim, latent_dim)
        
        # Projection matrices
        self.W_q = nn.Linear(input_dim, self.num_heads * self.hidden_dim)
        self.W_k = nn.Linear(latent_dim, self.num_heads * self.hidden_dim)
        self.W_v = nn.Linear(latent_dim, self.num_heads * self.hidden_dim)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        # Output projection
        self.W_o = nn.Linear(num_heads * self.hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        # Rotary embedding
        self.use_rotary_emb = use_rotary_emb
    
    def _seq_mask(
        self,
        scores,
        valid_lens: torch.Tensor
    ):
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
        z,
        valid_lens=None
    ):
        # Latent representation for key, value
        if z is None:  # For training
            z = self.latent(x)
        else:  # For KV-cache during inference
            z = torch.cat([z, self.latent(x)], dim=1)  # Stack along sequence dimension
        
        # Reshape for projection
        batch_size, seq_len, kv_len = x.size(0), x.size(1), z.size(1)
        
        # Projection
        query = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)
        key = self.W_k(z).view(batch_size, kv_len, self.num_heads, self.hidden_dim).transpose(1, 2)
        value = self.W_v(z).view(batch_size, kv_len, self.num_heads, self.hidden_dim).transpose(1, 2)
        
        if self.use_rotary_emb:
            query = rotary_emb(query, torch.arange(seq_len, device=query.device))
            key = rotary_emb(key, torch.arange(kv_len, device=key.device))

        # Compute attention for all heads at once
        attn_weights, scores = self.attention(query, key)
        head_outputs = torch.matmul(self._seq_mask(scores, valid_lens), value)  # Shape: (batch_size, num_heads, seq_len, hidden_dim)
        
        # Reshape and concatenate head outputs
        head_outputs = head_outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, num_heads * hidden_dim)

        output = self.W_o(head_outputs)
        output = self.dropout(output)
        
        return output, z, attn_weights, scores


# Unit test
if __name__ == "__main__":
    attention = MultiHeadLatentAttention(8, 16, 2, use_rotary_emb=True)
    x = torch.ones((1, 3, 8))
    z = None
    output, z, attn_weights, scores = attention(x, z)
    print("test 1")
    print(z.shape)
    print(output.shape)
    print(attn_weights.shape)
    print(scores.shape)

    # Continue stacking z
    print("test 2")
    output_2, z, attn_weights_2, scores_2 = attention(x, z) 
    print(z.shape)
    print(output_2.shape)
    print(attn_weights_2.shape)
    print(scores_2.shape)

    # Test valid lens - autoreg masked
    print("test 3")
    output_3, z, attn_weights_3, scores_3 = attention(x, None, valid_lens=torch.arange(3, device=x.device).unsqueeze(0))
    print(z.shape)
    print(output_3.shape)
    print(attn_weights_3)
    print(scores_3)