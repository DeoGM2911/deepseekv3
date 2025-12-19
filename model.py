import torch
from torch import nn
from deepseek_utils import Decoder
from dataclasses import dataclass


@dataclass(init=True, match_args=True)
class ModelConfig:
    vocab_size: int = 1000
    num_layers: int = 27
    num_dense_layers: int = 3
    input_dim: int = 4096
    kv_lora_dim: int = 576
    q_lora_dim: int = 576
    qk_rope_dim: int = 128
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    num_heads: int = 128
    indexer_num_heads: int = 64
    indexer_head_dim: int = 32
    indexer_rope_dim: int = 16
    indexer_k: int = 256
    max_len: int = 10000
    moe_hidden_dim: int = 4096
    moe_num_shared_experts: int = 16
    moe_num_experts: int = 256
    moe_k: int = 64
    moe_router: str = 'linear'


class DeepSeekV3(nn.Module):
    """
    The DeepSeek V3.2 architecture. This architecture
    introduces the Multi-head Latent Attention (MLA) and the DeepSeek Sparse
    Attention (DSA) mechanisms that is built on top of MLA.

    For more info, please read the series of papers on DeepSeek V3.x.

    @misc{deepseekai2024deepseekv32,
      title={DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention}, 
      author={DeepSeek-AI},
      year={2025},
    }
    """
    def __init__(
        self,
        config: ModelConfig
    ):
        super(DeepSeekV3, self).__init__()
        self.config = config

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.input_dim)

        # Decoder layers
        self.decoders = nn.ModuleList([
            Decoder(
                i,
                config.num_dense_layers,
                config.input_dim,
                config.kv_lora_dim,
                config.q_lora_dim,
                config.qk_rope_dim,
                config.qk_nope_head_dim,
                config.v_head_dim,
                config.num_heads,
                config.indexer_num_heads,
                config.indexer_head_dim,
                config.indexer_rope_dim,
                config.indexer_k,
                config.max_len,
                config.moe_hidden_dim,
                config.moe_num_shared_experts,
                config.moe_num_experts,
                config.moe_k,
                config.moe_router
            )
            for i in range(config.num_layers)
        ])

        # Output layer
        self.output = nn.Linear(config.input_dim, config.vocab_size)

    def forward(
        self,
        x,
        cache=False,
        kv=None,
        k_rope=None,
        k_idx=None,
        mask=None,
        inference=False
    ):
        # Embedding
        x = self.embedding(x)
        
        # Cache the input if needed
        new_kv = None
        new_k_rope = None
        new_k_idx = None
        if cache:
            new_kv = []
            new_k_rope = []
            new_k_idx = []
        
        # If the cache is None, create a temporary None list as inputs
        kv_list = [None] * self.config.num_layers if kv is None else kv
        k_rope_list = [None] * self.config.num_layers if k_rope is None else k_rope
        k_idx_list = [None] * self.config.num_layers if k_idx is None else k_idx

        # Remember the logit for training
        if not inference:
            gate_logits = []
            topk_indices = []
            attn_weights = []
            unmasked_scores = []
            indexer_indices = []
            indexer_scores = []
        else:
            gate_logits = None
            topk_indices = None
            attn_weights = None
            unmasked_scores = None
            indexer_indices = None
            indexer_scores = None

        # Decoder
        for i, decoder in enumerate(self.decoders):
            x, blk_kv, blk_k_rope, blk_k_idx, blk_attn_weights, \
                blk_unmasked_scores, blk_indexer_indices, blk_indexer_scores, \
                blk_gate_logits, blk_topk_indices \
            = decoder(x, kv_list[i], k_rope_list[i], k_idx_list[i], mask)

            # For calculate auxiliary loss
            if not inference:
                attn_weights.append(blk_attn_weights)
                unmasked_scores.append(blk_unmasked_scores)
                indexer_indices.append(blk_indexer_indices)
                indexer_scores.append(blk_indexer_scores)
                # Only collect for MoE layers
                if i > self.config.num_dense_layers:
                    gate_logits.append(blk_gate_logits)
                    topk_indices.append(blk_topk_indices)

            if cache:
                new_kv.append(blk_kv)
                new_k_rope.append(blk_k_rope)
                new_k_idx.append(blk_k_idx)
        
        # Convert the cache to tensors
        if cache:
            new_kv = torch.stack(new_kv, dim=0)
            new_k_rope = torch.stack(new_k_rope, dim=0)
            new_k_idx = torch.stack(new_k_idx, dim=0)
        
        # Output
        x = self.output(x)
        return x, new_kv, new_k_rope, new_k_idx, attn_weights, unmasked_scores, \
            indexer_indices, indexer_scores, gate_logits, topk_indices