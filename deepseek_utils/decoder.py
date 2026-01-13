import torch
from torch import nn
from .attention import MLA
from .moe import MoE, MLP
from .utils import RMSNorm


class Decoder(nn.Module):
    """
    The decoder in DeepSeek v3.2.
    A decoder block consists of a MLA layer, a MoE layer, and 
    2 residual pre-layer norm connection.

    ## Global Param

    @Args:
        block_idx: the index of the block
        num_dense_layers: the total number of layers with dense MLP blocks.

    ## MLA params:
    
    @Args:
        input_dim: the dimension of the input tensor
        kv_lora_dim: the dimension of the latent kv_cache
        q_lora_dim: the dimension of the latent query
        qk_rope_dim: the dimension of the latent query and key for RoPE
        qk_nope_head_dim: the dimension of the latent query and key for no RoPE
        v_head_dim: the dimension of the latent value
        num_heads: how many heads to use in the MLA
        indexer_num_heads: how many heads to use in the indexer
        indexer_head_dim: the dimension of the projected keys and queries in the indexer
        indexer_rope_dim: the dimension of the projected keys and queries for RoPE in the indexer
        indexer_k: how many keys to choose in the indexer
        max_len: the maximum length of the input sequence
    ## MoE params:
    
    @Args:
        input_dim = output_dim: the dimension of the input and output tensor
        moe_hidden_dim: the dimension of the hidden layer in the MoE
        moe_num_shared_experts: the number of shared experts in the MoE
        moe_num_experts: the number of experts in the MoE
        moe_k: how many experts to use in the MoE
        moe_router: the router to use in the MoE
    """
    def __init__(
        self,
        block_idx,
        num_dense_layers,
        input_dim,
        kv_lora_dim,
        q_lora_dim,
        qk_rope_dim,
        qk_nope_head_dim,
        v_head_dim,
        num_heads,
        indexer_num_heads,
        indexer_head_dim,
        indexer_rope_dim,
        indexer_k,
        max_len,
        moe_hidden_dim,
        moe_num_shared_experts,
        moe_num_experts,
        moe_k,
        moe_router
    ):
        super(Decoder, self).__init__()
        self.block_idx = block_idx
        self.use_moe = self.block_idx >= num_dense_layers

        # MLA layer
        self.mla = MLA(
            input_dim,
            kv_lora_dim,
            q_lora_dim,
            qk_rope_dim,
            qk_nope_head_dim,
            v_head_dim,
            num_heads,
            indexer_num_heads,
            indexer_head_dim,
            indexer_rope_dim,
            indexer_k,
            max_len
        )

        # MoE layer
        if self.use_moe:
            self.moe = MoE(
                input_dim,
                moe_hidden_dim,
                input_dim,
                moe_num_shared_experts,
                moe_num_experts,
                moe_k,
                moe_router
            )
        else:
            self.moe = MLP(input_dim, moe_hidden_dim, input_dim)

        # Residual pre-Layer Norm
        self.ln1 = RMSNorm(input_dim)
        self.ln2 = RMSNorm(input_dim)

    def forward(
        self,
        x,
        kv=None,
        k_rope=None,
        k_idx=None,
        mask=None
    ):
        #  Prelayer Norm
        x_norm = self.ln1(x)
        # MLA
        x_mla, kv, k_rope, k_idx, attn_weights, unmasked_scores, indexer_indices, indexer_scores = \
            self.mla(x_norm, kv, k_rope, k_idx, mask)
        # Residual connection
        x = x_mla + x_norm
        
        # Pre-layer norm
        x_norm = self.ln2(x)
        # MoE
        if self.use_moe:
            x_moe, gate_logits, topk_indices = self.moe(x_norm)        
        else:
            x_moe = self.moe(x_norm)
            gate_logits, topk_indices = None, None
        # Residual connection
        x = x_norm + x_moe

        return x, kv, k_rope, k_idx, attn_weights, unmasked_scores, \
                indexer_indices, indexer_scores, gate_logits, topk_indices
