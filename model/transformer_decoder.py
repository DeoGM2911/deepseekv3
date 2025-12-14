import torch
from torch import nn
from .attention import MultiHeadLatentAttention
from .MoE import MixtureOfExperts, PositionalFFN


class DecoderBlock(nn.Module):
    """
    A single decoder block in a transformer decoder with MLA and MoE/FFN. For details about MoE
    and MLA, please see the documentation of the corresponding modules.
    """
    def __init__(
        self,
        input_dim,
        latent_dim,
        num_heads,
        num_experts,
        ffn_hidden_dim,
        moe=True,
        k=1,
        router="centroid",
        dropout=0.1
    ):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadLatentAttention(input_dim, latent_dim, num_heads, dropout)
        if moe:
            self.moe = MixtureOfExperts(input_dim, input_dim, num_experts, ffn_hidden_dim, k, router, dropout)
        else:
            self.moe = PositionalFFN(input_dim, input_dim, ffn_hidden_dim, dropout)
        self.add_norm1 = nn.RMSNorm(input_dim)
        self.add_norm2 = nn.RMSNorm(input_dim)
    
    def forward(
        self,
        x,
        z=None,
        key_rope=None,
        valid_lens=None,
        inference=False
    ):
        # Norm
        x_norm = self.add_norm1(x)

        # Self-attention
        attn_output, new_z, new_key_rope, attn_weights, scores = self.attention(x_norm, z, key_rope, valid_lens, inference)
        # Add residual connection
        x = x + attn_output
        # Norm
        x_norm = self.add_norm2(x) 
        # MoE
        moe_output, moe_logits, moe_topk_indices = self.moe(x_norm)
        # Add residual connection
        output = x + moe_output
        
        return output, attn_weights, scores, moe_logits, moe_topk_indices, new_z, new_key_rope


# Unit test
if __name__ == "__main__":
    blk = DecoderBlock(8, 16, 2, 2, 16)
    x = torch.randn(2, 3, 8)
    z = torch.randn(2, 2, 16)
    key_rope = torch.randn(2, 2, 16)
    valid_lens = torch.tensor([[0, 1, 2], [0, 1, 2]])
    output, attn_weights, scores, moe_logits, moe_topk_indices, new_z, new_key_rope = blk(x, z, key_rope, valid_lens)
    print(output.shape)
    print(attn_weights.shape)
    print(scores.shape)
    print(moe_logits.shape)
    print(moe_topk_indices.shape)
    print(new_z.shape)
    print(new_key_rope.shape)
    