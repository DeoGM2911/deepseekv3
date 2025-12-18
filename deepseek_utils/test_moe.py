import torch
from moe import MoE


def test_moe():
    moe = MoE(
        input_dim=4,
        hidden_dim=16, 
        output_dim=4, 
        num_shared_experts=2, 
        num_experts=6, 
        k=3
    )
    moe.eval()
    print("Linear router")
    o, logits, _ = moe(torch.randn(3, 6, 4))
    print(o.shape)
    print(logits.shape)

    moe = MoE(
        input_dim=4,
        hidden_dim=16, 
        output_dim=4, 
        num_shared_experts=2, 
        num_experts=6, 
        k=3,
        router="centroid"
    )
    moe.eval()
    print("Centroid router")
    o, logits, _ = moe(torch.randn(3, 6, 4))
    print(o.shape)
    print(logits.shape)


if __name__ == "__main__":
    test_moe()