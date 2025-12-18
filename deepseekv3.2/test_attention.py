import torch
from attention import ScaledDotProductAttention, Indexer


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
    mask_2 = torch.randint(0, 2, (2, 3, 4))
    indexer.eval()
    top_indices_1, top_scores_1, _ = indexer(x, kv_1, mask_1)
    top_indices_2, top_scores_2, _ = indexer(x, kv_2, mask_2)
    print(mask_1)
    print("Less the k choices")
    print(top_indices_1)
    print(top_scores_1)
    print("More the k choices")
    print(mask_2)
    print(top_indices_2)
    print(top_scores_2)


if __name__ == "__main__":
    # print("Test dot product attention")
    # test_dot_prod_attention()
    print("Test indexer")
    test_indexer()