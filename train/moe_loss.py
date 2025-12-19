import torch
import torch.nn.functional as F


def compute_load_balancing_loss(gate_logits, topk_indices, num_experts):
    """
    Compute auxiliary loss to encourage balanced expert usage.
    
    Args:
        gate_logits: (num_layers, batch, seq_len, num_experts) - raw gating scores
        topk_indices: (num_layers, batch, seq_len, k) - indices of selected experts
        num_experts: total number of experts
    
    Returns:
        load_balance_loss: scalar tensor
    """
    num_layers, batch_size, seq_len, _ = gate_logits.shape
    
    # 1. Compute fraction of tokens routed to each expert
    # Create one-hot encoding of selected experts
    expert_mask = torch.zeros_like(gate_logits)  # (num_layers, batch, seq_len, num_experts)
    expert_mask.scatter_(-1, topk_indices, 1.0)
    
    # Fraction of tokens assigned to each expert
    f_i = expert_mask.sum(dim=[1, 2]) / (batch_size * seq_len)  # (num_layers, num_experts)
    
    # 2. Compute average gate probability for each expert
    gate_probs = F.softmax(gate_logits, dim=-1)  # (num_layers, batch, seq_len, num_experts)
    P_i = gate_probs.mean(dim=[1, 2])  # (num_layers, num_experts)
    
    # 3. Load balancing loss: encourages f_i and P_i to be uniform
    # If perfectly balanced, both would be 1/num_experts
    loss = num_experts * torch.mean(torch.sum(f_i * P_i, dim=-1))
    
    return loss