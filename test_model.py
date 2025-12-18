import torch
from model import DeepSeekV3, ModelConfig


def test_model():
    config = ModelConfig(
        vocab_size = 10,
        num_layers = 3,
        num_dense_layers = 1,
        input_dim = 32,
        kv_lora_dim = 16,
        q_lora_dim = 16,
        qk_rope_dim = 8,
        qk_nope_head_dim = 8,
        v_head_dim = 8,
        num_heads = 2,
        indexer_num_heads = 2,
        indexer_head_dim = 8,
        indexer_rope_dim = 4,
        indexer_k = 2,
        max_len = 1024,
        moe_hidden_dim = 32,
        moe_num_shared_experts = 4,
        moe_num_experts = 8,
        moe_k = 2,
        moe_router = 'linear',
        moe_dropout = 0.1
    )

    model = DeepSeekV3(config)
    
    # Dummy data
    batch_size = 3
    seq_len = 10
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    print("Test no cache, training")
    mask = torch.randint(0, 2, (batch_size, seq_len, seq_len))
    o, kv, k_rope, k_idx, attn_weights, kv_mask, gate_logits, topk_indices = model(x, mask=mask)
    print(torch.stack(gate_logits, dim=0).shape)
    print(torch.stack(topk_indices, dim=0).shape)

    print("Test cache, prefilling")
    mask = torch.randint(0, 2, (batch_size, seq_len, seq_len))
    o, kv, k_rope, k_idx, attn_weights, kv_mask, gate_logits, topk_indices = model(x, cache=True, mask=mask, inference=True)
    print(kv.shape)
    print(k_rope.shape)
    print(k_idx.shape)

    print("Test cache, decoding with cache")
    x = torch.randint(0, config.vocab_size, (batch_size, 1))
    for i in range(3):
        print(f"Step {i}")
        o, kv, k_rope, k_idx, attn_weights, kv_mask, gate_logits, topk_indices = \
            model(x, cache=True, kv=kv, k_rope=k_rope, k_idx=k_idx, inference=True)
        print(kv.shape)
        print(k_rope.shape)
        print(k_idx.shape)


if __name__ == "__main__":
    test_model()
