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
        rotary_enc=True,
        router="centroid",
        dropout=0.1
    ):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadLatentAttention(input_dim, latent_dim, num_heads, rotary_enc, dropout)
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
        valid_lens=None,
    ):
        # Norm
        x_norm = self.add_norm1(x)

        # Self-attention
        attn_output, new_z, attn_weights, scores = self.attention(x_norm, z, valid_lens)
        # Add residual connection
        x = x + attn_output
        # Norm
        x_norm = self.add_norm2(x) 
        # MoE
        moe_output, moe_logits, moe_topk_indices = self.moe(x_norm)
        # Add residual connection
        output = x + moe_output
        
        return output, attn_weights, scores, moe_logits, moe_topk_indices, new_z
