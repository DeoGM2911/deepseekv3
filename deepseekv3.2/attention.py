import torch
from torch import nn
from torch.nn import functional as F
import math
from utils import rotary_emb, pre_compute_theta


class ScaledDotProductAttention(nn.Module):
    """
    The scaled dot-product attention mechanism for Multi-head Attention.
    
    Given a query matrix Q and a key matrix K, this module computes the attention scores
    as the dot product of Q and K, scaled by the square root of the dimensionality of the key vectors.
    The scores are then passed through a softmax function to obtain attention weights and optionally
    applies dropout for regularization.

    Note that Q and K should have shape (B, H, T, D) where H is the number of heads, T is num queries
    or num keys, and D is the dimensionality of the key vectors (which should be the same as that of
    the keys).

    \\[
        \\operatorname{softmax}\\bigg(\\dfrac{QK^T}{\\sqrt{d_k}}\\bigg)
    \\]

    One can provide a mask for the attention weights to control the flow of information.
    Note that the expected format is (batch_size, num_queries, num_keys) and the entries are binary.
    1 means keep, 0 means ignore.

    This allows for the incorporation of both causal and pad masking. In training, expect
    num_queries = num_keys = seq_len. In inference, num_queries = 1.

    Also note that we can combine this mask with other mask (e.g. Indexer mask) for more complex
    masking.
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
        mask
    ):
        """
        Perform scaled dot product attention. Please read the class doc string
        to see the requirement for mask. 
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        return attn_weights, scores


class Indexer(nn.Module):
    """
    The indexer in DeepSeek Sparse Attention.
    Compute the relevant scores for each keys/values in the kv_cache and decide which keys/values 
    to use in the attention mechanism.

    @Args:
        input_dim: the dimension of the input tensor
        kv_lora_dim: the dimension of the latent kv_cache
        num_heads: how many heads to use in the indexer
        head_dim: the dimension of the projected keys and queries in the indexer
        k: how many keys to choose
    """
    def __init__(
        self,
        input_dim,
        kv_lora_dim,
        num_heads,
        head_dim,
        rope_dim,
        k,
        max_len
    ):
        super(Indexer, self).__init__()
        self.input_dim = input_dim
        self.kv_lora_dim = kv_lora_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.k = k
        self.max_len = max_len

        # Precompute theta for RoPE
        self.theta = pre_compute_theta(max_len, self.rope_dim)

        # Projection matrices
        self.Wq_idx = nn.Linear(self.input_dim, self.num_heads * self.head_dim)
        self.Wk_idx = nn.Linear(self.kv_lora_dim, self.head_dim)

        # Weights
        self.W_weights = nn.Linear(self.input_dim, self.num_heads)

        # Norms
        self.q_norm = nn.LayerNorm(self.num_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.w_norm = nn.LayerNorm(self.num_heads)

    def forward(
        self,
        x,
        kv,
        mask=None,
        k_idx=None  # cache for k_idx,
    ):
        """
        Compute the relevant scores to decide which keys/values should the attention mechanism
        focus on. 
        
        Args:
            x: Query matrix of shape (B, H, T, D)
            kv: KV-lora matrix of shape (B, H, T, D)
            mask: The mask for the indexer. Should be the mask that will be use in the attention mechanism.
                    Should be of shape (B, T, KV_len) in training.
            k_idx: Cache for k_idx. Should be None for training or at first inference call.

        Precaution: One need to apply the mask again before choosing the top-k indices to ensure
        causality.
        """
        batch_size, seq_len, _ = x.shape

        # Compute the keys for the indexer. Cache if needed
        if k_idx is None:
            k_idx = self.k_norm(self.Wk_idx(kv))
        else:
            k_idx = torch.cat([k_idx, self.k_norm(self.Wk_idx(kv))], dim=1)
        
        kv_len = k_idx.shape[1]

        # Compute the queries for the indexer
        q_idx = self.q_norm(self.Wq_idx(x))
        
        # Compute the weights for the indexer
        w_idx = self.w_norm(self.W_weights(x))

        # Apply rope for keys and queries
        k_nope, k_rope = k_idx.split([self.rope_dim, self.head_dim-self.rope_dim], dim=-1)
        q_nope, q_rope = q_idx.split([self.num_heads * self.rope_dim, self.num_heads * (self.head_dim-self.rope_dim)], dim=-1)

        # Apply RoPE
        q_rope = q_rope.view(batch_size, seq_len, self.num_heads, self.rope_dim)
        k_rope = k_rope.view(batch_size, kv_len, 1, self.rope_dim)
        k_rope = rotary_emb(k_rope, self.theta)
        q_rope = rotary_emb(q_rope, self.theta)

        # Reconcat
        k_idx = torch.cat([k_nope.view(batch_size, kv_len, 1, self.head_dim-self.rope_dim), k_rope], dim=-1)
        q_idx = torch.cat([q_nope.view(batch_size, seq_len, self.num_heads, self.head_dim-self.rope_dim), q_rope], dim=-1)

        # Reshape for the similarity scores
        k_idx = k_idx.view(batch_size, 1, 1, kv_len, -1)
        q_idx = q_idx.view(batch_size, seq_len, self.num_heads, 1, -1)
        w_idx = w_idx.view(batch_size, seq_len, self.num_heads, 1)

        # Compute the scores
        idx_scores = w_idx * F.relu((k_idx * q_idx).sum(dim=-1))  # (batch_size, seq_len, num_heads, kv_len)
        idx_scores = idx_scores.sum(-2)  # (batch_size, seq_len, kv_len)

        # Apply mask
        if mask is not None:
            idx_scores = idx_scores.masked_fill(mask == 0, -1e9)
        
        # Choose top-k keys/values
        top_scores, top_indices = idx_scores.topk(min(self.k, kv_len), dim=-1)
        return top_indices, idx_scores, k_idx


class MLA(nn.Module):
    """
    The MLA in DeepSeek Sparse Attention.
    """
    def __init__(
        self,
        input_dim,
        kv_lora_dim,
        q_lora_dim,
        qk_rope_dim,
        qk_head_dim,
        v_head_dim,
        num_heads,
        indexer_num_heads,
        indexer_head_dim,
        indexer_k,
        max_len
    ):
        super(MLA, self).__init__()

        self.input_dim = input_dim
        self.kv_lora_dim = kv_lora_dim
        self.q_lora_dim = q_lora_dim
        self.qk_rope_dim = qk_rope_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.num_heads = num_heads
        self.indexer_num_heads = indexer_num_heads
        self.indexer_head_dim = indexer_head_dim
        self.indexer_k = indexer_k

        # Down projection matrices
        self.Wq_a = nn.Linear(self.input_dim, self.q_lora_dim)
        self.Wkv_a = nn.Linear(self.input_dim, self.kv_lora_dim + self.num_heads * self.qk_rope_dim)

        # Norm layers
        self.q_norm = nn.LayerNorm(self.num_heads * (self.qk_head_dim + self.qk_rope_dim))
        self.kv_norm = nn.LayerNorm(self.kv_lora_dim + self.num_heads * self.qk_rope_dim)
        self.k_norm = nn.LayerNorm(self.qk_head_dim)
        self.v_norm = nn.LayerNorm(self.v_head_dim)

        # Up projection matrices
        self.Wq_b = nn.Linear(self.q_lora_dim, self.num_heads * (self.qk_head_dim + self.qk_rope_dim))
        self.Wkv_b = nn.Linear(self.kv_lora_dim, self.num_heads * (self.qk_head_dim + self.v_head_dim))
        
        # Precompute the theta angle for RoPE 
        self.m_theta = pre_compute_theta(max_len, self.qk_rope_dim)

        # Indexer
        self.indexer = Indexer(
            input_dim=self.input_dim,
            kv_lora_dim=self.kv_lora_dim,
            num_heads=self.indexer_num_heads,
            head_dim=self.indexer_head_dim,
            k=self.indexer_k
        )

        # Output projection matrix
        self.W_o = nn.Linear(self.v_head_dim, self.input_dim)
    
    def forward(
        self,
        x,
        kv_cache=None,
        k_rope=None,
        mask=None,
        k_idx=None
    ):
        """
        Perform the DSA-MLA attention. There are 2 modes:
        - MHA-MLA: use for prefilling/training 
        - MQA-MLA: use for decoding/inference

        @Args:
            x: the input tensor
            kv_cache: the cache for kv latent
            k_rope: the cache for key rope projection
            mask: the mask for the attention mechanism
            k_idx: the cache for indexer's keys
        """
        # Compute the kv latent & key rope projection. Cache if needed
        cache = self.kv_norm(self.Wkv_a(x)) 
        if kv_cache is None:
            kv_cache, k_rope = cache.split([self.kv_lora_dim, self.num_heads * self.qk_rope_dim], dim=-1)
        else:
            cache, rope = kv_cache.split([self.kv_lora_dim, self.num_heads * self.qk_rope_dim], dim=-1)
            kv_cache = torch.cat([kv_cache, cache], dim=1)
            k_rope = torch.cat([k_rope, rope], dim=1)
        
        # Compute the queries for the attention mechanism
        q = self.Wq_a(x)
        q = self.q_norm(self.Wq_b(q))
        q_nope, q_rope = q.split([self.num_heads * self.qk_head_dim, self.num_heads * self.qk_rope_dim], dim=-1)

        ...