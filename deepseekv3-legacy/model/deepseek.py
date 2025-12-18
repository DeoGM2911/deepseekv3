import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import PosEncoding
from .decoder import DecoderBlock


class DeepSeekV3Config:
    """
    Config class for DeepSeek V3
    """
    vocab_size=100
    input_dim=4096
    latent_dim=576
    max_len=1000
    num_layers=61
    num_heads=128
    k=8
    ffn_hidden_dim=4096
    num_experts=256
    router="centroid"
    dropout=0.1
    mode_switch=30
    num_ffn=3
    indexer_num_heads=16
    indexer_dim=32
    indexer_k=16


class AnswerHead(nn.Module):
    """
    A simple linear layer for answer head (LM head).
    """
    def __init__(
        self,
        input_dim,
        vocab_size
    ):
        super(AnswerHead, self).__init__()
        self.lm_head = nn.Linear(input_dim, vocab_size)
    
    def forward(self, x):
        return self.lm_head(x)


class DeepSeekV3(nn.Module):
    """
    DeepSeek V3 architecture.

    @Args:
    vocab_size: int
        The size of the vocabulary.
    input_dim: int
        The dimension of the input.
    latent_dim: int
        The dimension of the latent space.
    max_len: int
        The maximum length of the input sequence.
    num_layers: int
        The number of decoder blocks.
    num_heads: int
        The number of attention heads.
    moe: bool
        Whether to use MoE.
    k: int
        The number of experts to use.
    ffn_hidden_dim: int
        The dimension of the hidden layer in the FFN.
    num_experts: int
        The number of experts in MoE.
    router: str
        The router to use.
    dropout: float
        The dropout rate.
    mode_switch: int
        The number of decoder blocks to use MHA.
    num_ffn: int
        The number of initial layer to use FFN.
    """
    def __init__(
        self,
        vocab_size,
        input_dim,
        latent_dim=576,
        max_len=1000,
        num_layers=8,
        num_heads=8,
        k=8,
        ffn_hidden_dim=1024,
        num_experts=16,
        router="centroid",
        dropout=0.1,
        mode_switch=4,
        num_ffn=3
    ):
        super(DeepSeekV3, self).__init__()

        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, input_dim)
        
        # First num_ffn decoder blocks use normal FFN
        num_dense = min(num_ffn, num_layers)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                input_dim,
                latent_dim,
                num_heads,
                num_experts,
                ffn_hidden_dim,
                moe=False,
                use_indexer=True,
                indexer_num_heads=16,
                indexer_dim=32,
                indexer_k=16,
                k=k,
                router=router,
                dropout=dropout,
                mode="MHA" if mode_switch >= idx else "MQA"
            )
            for idx in range(num_dense)
        ])

        # The remaining decoder blocks use MoE
        self.decoder_blocks.extend(nn.ModuleList([
            DecoderBlock(
                input_dim,
                latent_dim,
                num_heads,
                num_experts,
                ffn_hidden_dim,
                moe=True,  # Use parameter
                use_indexer=True,
                indexer_num_heads=16,
                indexer_dim=32,
                indexer_k=16,
                k=k,      # Use parameter
                router=router,  # Use parameter
                dropout=dropout,  # Use parameter
                mode="MHA" if mode_switch >= idx else "MQA"
            )
            for idx in range(num_layers - num_dense)
        ]))

        # The answer head (LM head) - uses input_dim, not latent_dim
        self.answer_head = AnswerHead(input_dim, vocab_size)
    
    def _fuse_q_projection(self):
        """
        Fuse all query projection matrices after training.

        WARNING: ONLY CALL this after training during inference.
        """
        for block in self.decoder_blocks:
            block.attention._fuse_q_projection()

    def forward(
        self,
        x,
        cache=False,
        kv_cache=None,
        key_rope=None,
        attention_mask=None,
        inference=False
    ):
        # KV cache
        if cache:
            kv_cache_memory = []
            key_rope_memory = []
        else:
            kv_cache_memory = None
            key_rope_memory = None
        
        if kv_cache is None:
            kv_cache = [None] * self.num_layers
        else:
            # Unpack tensor to list for iteration
            kv_cache = list(kv_cache.unbind(dim=0))
        
        if key_rope is None:
            key_rope = [None] * self.num_layers
        else:
            # Unpack tensor to list for iteration
            key_rope = list(key_rope.unbind(dim=0))
        
        # MoE logits for auxilary loss
        moe_logits_list = []
        moe_topk_indices_list = []

        # Embedding
        x = self.embedding(x)

        # Decoder blocks - capture all return values
        for i, block in enumerate(self.decoder_blocks):
            x, _, _, moe_logits, moe_topk_indices, new_kv_cache, new_key_rope = block(x, kv_cache[i], key_rope[i], attention_mask, inference)
            
            if cache:
                kv_cache_memory.append(new_kv_cache)
                key_rope_memory.append(new_key_rope)
            
            moe_logits_list.append(moe_logits) if moe_logits is not None else None
            moe_topk_indices_list.append(moe_topk_indices) if moe_topk_indices is not None else None
        
        if cache and kv_cache_memory is not None:
            kv_cache_tensor = torch.stack(kv_cache_memory, dim=0)
            key_rope_tensor = torch.stack(key_rope_memory, dim=0)
        else:
            kv_cache_tensor = None
            key_rope_tensor = None

        # Stack the moe logits and indices
        moe_logits_list = torch.stack(moe_logits_list, dim=0) if moe_logits_list else None
        moe_topk_indices_list = torch.stack(moe_topk_indices_list, dim=0) if moe_topk_indices_list else None

        # Answer head
        return self.answer_head(x), (moe_logits_list, moe_topk_indices_list), kv_cache_tensor, key_rope_tensor


if __name__ == "__main__":
    dummy_vocab = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9}
    dummy_input = torch.randint(0, 10, (3, 10))  # batch_size=2
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    input_dim = 256
    
    # Create model
    deepseek = DeepSeekV3(
        vocab_size=len(dummy_vocab),
        input_dim=input_dim,
        num_layers=8
    )
    
    deepseek._fuse_q_projection()

    # Test forward pass
    output, _, kv_cache, key_rope = deepseek(dummy_input, cache=True, kv_cache=None, key_rope=None, attention_mask=attention_mask, inference=False)
    print(f"Output shape: {output.shape}")  # Should be (3, 10, vocab_size)
    print(f"kv_cache shape: {kv_cache.shape}")
    print(f"key_rope shape: {key_rope.shape}")

    # Test inference mode
    dummy_input_2 = torch.randint(0, 10, (3, 1))
    attention_mask = torch.cat([attention_mask, torch.zeros((3, 1))], dim=1)
    output, _, kv_cache, key_rope = deepseek(dummy_input_2, cache=True, kv_cache=kv_cache, key_rope=key_rope, attention_mask=attention_mask, inference=True)
    print(f"Output shape: {output.shape}")  # Should be (3, 1, vocab_size)
    print(f"kv_cache shape: {kv_cache.shape}")
    print(f"key_rope shape: {key_rope.shape}")

    # Test inference mode
    dummy_input_3 = torch.randint(0, 10, (3, 1))
    attention_mask = torch.cat([attention_mask, torch.zeros((3, 1))], dim=1)
    output, _, kv_cache, key_rope = deepseek(dummy_input_3, cache=True, kv_cache=kv_cache, key_rope=key_rope, attention_mask=attention_mask, inference=True)
    print(f"Output shape: {output.shape}")  # Should be (3, 1, vocab_size)
    print(f"kv_cache shape: {kv_cache.shape}")
    print(f"key_rope shape: {key_rope.shape}")
