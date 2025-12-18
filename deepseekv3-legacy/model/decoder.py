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
        use_indexer=True,
        indexer_num_heads=8,
        indexer_dim=64,
        indexer_k=4,
        router="centroid",
        mode="MHA",
        dropout=0.1
    ):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadLatentAttention(
                                input_dim,
                                latent_dim,
                                num_heads,
                                use_indexer,
                                indexer_num_heads,
                                indexer_dim,
                                indexer_k,
                                dropout,
                                mode
                            )
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
        attention_mask=None,
        inference=False
    ):
        # Norm
        x_norm = self.add_norm1(x)

        # Self-attention
        attn_output, new_z, new_key_rope, attn_weights, scores = self.attention(x_norm, z, key_rope, attention_mask, inference)
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
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    print("Test 1")
    output, attn_weights, scores, moe_logits, moe_topk_indices, new_z, new_key_rope = blk(x, None, None, None, inference=False)
    print(output.shape)
    print(attn_weights.shape)
    print(scores.shape)
    print(moe_logits.shape)
    print(moe_topk_indices.shape)
    print(new_z.shape)
    print(new_key_rope.shape)

    x = torch.randn(2, 1, 8)
    attention_mask = torch.tensor([[1, 1, 0, 1], [1, 0, 0, 1]])
    print("Test 2")
    output, attn_weights, scores, moe_logits, moe_topk_indices, new_z, new_key_rope = blk(x, new_z, new_key_rope, attention_mask, inference=True)
    print(output.shape)
    print(attn_weights.shape)
    print(scores.shape)
    print(moe_logits.shape)
    print(moe_topk_indices.shape)
    print(new_z.shape)
    print(new_key_rope.shape)

    blk = DecoderBlock(8, 16, 2, 2, 16, mode="MQA")
    x = torch.randn(2, 3, 8)
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    print("Test 3")
    output, attn_weights, scores, moe_logits, moe_topk_indices, new_z, new_key_rope = blk(x, None, None, None, inference=False)
    print(output.shape)
    print(attn_weights.shape)
    print(scores.shape)
    print(moe_logits.shape)
    print(moe_topk_indices.shape)
    print(new_z.shape)
    print(new_key_rope.shape)

    x = torch.randn(2, 1, 8)
    attention_mask = torch.tensor([[1, 1, 0, 1], [1, 0, 0, 1]])
    print("Test 4")
    output, attn_weights, scores, moe_logits, moe_topk_indices, new_z, new_key_rope = blk(x, new_z, new_key_rope, attention_mask, inference=True)
    print(output.shape)
    print(attn_weights.shape)
    print(scores.shape)
    print(moe_logits.shape)
    print(moe_topk_indices.shape)
    print(new_z.shape)
    print(new_key_rope.shape)
    