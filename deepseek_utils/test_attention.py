import torch
from attention import ScaledDotProductAttention, Indexer, MLA


def test_dot_prod_attention():
    attention = ScaledDotProductAttention()
    q, k = torch.ones((2, 4, 2, 8)), torch.rand((2, 4, 3, 8))  # (B, H, T, D)
    mask = torch.randint(0, 2, (2, 2, 3))
    attention.eval()
    # With mask
    attn_weights, scores = attention(q, k, mask)
    print(mask)
    print(attn_weights)


def test_indexer():
    indexer = Indexer(
        input_dim=8,
        kv_lora_dim=16,
        num_heads=4,
        head_dim=8,
        rope_dim=4,
        k=2,
        max_len=1024
    )
    x = torch.ones((2, 3, 8))
    kv_1 = torch.rand((2, 2, 16))
    kv_2 = torch.rand((2, 4, 16))
    mask_1 = torch.randint(0, 2, (2, 3, 2))
    indexer.eval()
    top_indices_1, top_scores_1, k_idx1 = indexer(x, kv_1, None, mask_1)
    top_indices_2, top_scores_2, k_idx2 = indexer(x, kv_2, k_idx1, None)
    print(mask_1)
    print("Less the k choices")
    print(top_indices_1)
    print(top_scores_1)
    print(k_idx1.shape)
    print("More the k choices")
    print(top_indices_2)
    print(top_scores_2)
    print(k_idx2.shape)


def test_mla():
    mla = MLA(
        input_dim=4,
        kv_lora_dim=8,
        q_lora_dim=4,
        qk_rope_dim=2,
        qk_nope_head_dim=4,
        v_head_dim=4,
        num_heads=2,
        indexer_num_heads=2,
        indexer_head_dim=4,
        indexer_rope_dim=2,
        indexer_k=2,
        max_len=1024
    )
    x = torch.ones((1, 3, 4))
    mask = torch.randint(0, 2, (1, 3, 3))
    mla.eval()
    print("Test prefilling/training")
    o, kv, k_rope, k_idx, _, kv_mask = mla(x, None, None, None, mask)
    print(kv)
    print(k_rope)
    print(k_idx)
    print(kv_mask)
    print(kv.shape)
    print(k_rope.shape)
    print(k_idx.shape)
    print(kv_mask.shape)
    print("----------------------------")
    print("Test decode/inference")
    x = torch.ones((1, 1, 4))
    for i in range(3):
        print(f"Step {i}")
        o, kv, k_rope, k_idx, _, kv_mask = mla(x, kv, k_rope, k_idx, None)
        print(kv.shape)
        print(k_rope.shape)
        print(k_idx.shape)
        print(kv_mask.shape)


if __name__ == "__main__":
    print("Test dot product attention")
    test_dot_prod_attention()
    print("Test indexer")
    test_indexer()
    print("Test MLA")
    test_mla()