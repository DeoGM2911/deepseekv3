import torch
from decoder import Decoder


def test_decoder():
    decoder = Decoder(
        block_idx=0,  # Change this to 3 to test with MoE instead of MLP
        num_dense_layers=2,
        input_dim=4,
        kv_lora_dim=8,
        q_lora_dim=4,
        qk_rope_dim=8,
        qk_nope_head_dim=4,
        v_head_dim=4,
        num_heads=2,
        indexer_num_heads=2,
        indexer_head_dim=4,
        indexer_rope_dim=2,
        indexer_k=2,
        max_len=1024,
        moe_hidden_dim=16,
        moe_num_shared_experts=1,
        moe_num_experts=4,
        moe_k=2,
        moe_router='linear'
    )

    x = torch.randn(2, 5, 4)
    mask = torch.randint(0, 2, (2, 5, 5))
    
    print("Prefill/Train")
    x, kv, k_rope, k_idx, attn_weights, kv_mask, gate_logits, topk_indices = decoder(x, mask=mask)
    print(kv.shape)
    print(k_rope.shape)
    print(k_idx.shape)
    print("Inference")
    x = torch.randn(2, 1, 4)
    x, kv, k_rope, k_idx, attn_weights, kv_mask, gate_logits, topk_indices = decoder(x, kv, k_rope, k_idx)
    print(kv.shape)
    print(k_rope.shape)
    print(k_idx.shape)
    x = torch.randn(2, 1, 4)
    x, kv, k_rope, k_idx, attn_weights, kv_mask, gate_logits, topk_indices = decoder(x, kv, k_rope, k_idx)
    print(kv.shape)
    print(k_rope.shape)
    print(k_idx.shape)


if __name__ == "__main__":
    test_decoder()