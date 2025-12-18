import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU)
    
    SwiGLU(x) = SiLU(xW_gate) ⊗ (xW_value)
    where SiLU(x) = x * sigmoid(x) (also known as Swish)
    """
    def __init__(
        self,
        input_dim,
        hidden_dim
    ):
        super(SwiGLU, self).__init__()
        # Two separate linear projections
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(input_dim, hidden_dim, bias=False)
    
    def forward(self, x):
        # Gate pathway: apply SiLU activation
        gate = F.silu(self.gate_proj(x))
        # Value pathway: linear projection
        value = self.value_proj(x)
        # Element-wise multiplication (gating)
        return gate * value


class CentroidRouter(nn.Module):
    """
    Centroid routing for MoE.

    Given a input x, we will compute the similarity between it and the centroids. In essence,
    the expert whose centroid is closest to x will handle x.
    """
    def __init__(
        self,
        input_dim,
        num_experts
    ):
        super(CentroidRouter, self).__init__()

        # The centroids
        self.centroids = nn.Parameter(torch.randn(num_experts, input_dim))
        self.temperature = nn.Parameter(torch.ones(1))
        self.num_experts = num_experts
        self.input_dim = input_dim
    
    def forward(
        self,
        x
    ):
        # Normalize the input and the centroids
        x = F.normalize(x, dim=-1)
        norm_centroids = F.normalize(self.centroids, dim=-1)

        # Compute the cosine similarity between the input and the centroids
        logits = torch.matmul(x, norm_centroids.T) / self.temperature
        return logits


class MixtureOfExperts(nn.Module):
    r"""
    A Mixture of Experts (MoE) layer that routes inputs to different expert networks
    based on learned gating mechanisms.
    
    This module consists of multiple expert networks and a gating network that determines
    the contribution of each expert to the final output.

    There is 1 shared expert and the are top k other experts which will be attended to.
    We provide two router types: linear gating and centroid gating. For DeepSeek V3, we use centroid gating.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        num_experts,
        hidden_dim,
        k,
        router="linear",
        dropout=0.1
    ):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.k = k  # number of top experts to use
        self.outptut_dim = output_dim

        # One shared expert - SwiGLU architecture
        self.shared_expert = nn.Sequential(
            SwiGLU(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        # The experts - SwiGLU architecture
        self.experts = nn.ModuleList([
            nn.Sequential(
                SwiGLU(input_dim, hidden_dim),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_experts)
        ])

        if router == "linear":
            self.router = nn.Linear(input_dim, num_experts)
        elif router == "centroid":
            self.router = CentroidRouter(input_dim, num_experts)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x
    ):
        batch_size, seq_len, _ = x.shape
        # Compute gating weights
        gate_logits = self.router(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # Choose top k experts
        topk_weights, topk_indices = torch.topk(gate_weights, self.k, dim=-1)
        gate_weights = torch.zeros_like(gate_weights).scatter_(-1, topk_indices, topk_weights)
        gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)  # Normalize
        
        # Compute expert outputs, ignore non-topk experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)  # Shape: (batch_size, seq_len, num_experts, output_dim)
        
        # Add a shared expert output
        expert_outputs = torch.cat([expert_outputs, self.shared_expert(x).unsqueeze(2)], dim=2)
        gate_weights = torch.cat([gate_weights, torch.ones(batch_size, seq_len, 1, device=x.device)], dim=-1)

        # Weighted sum of expert outputs
        gate_weights = gate_weights.unsqueeze(-1)  # Shape: (batch_size, seq_len, num_experts + 1, 1)
        output = torch.sum(gate_weights * expert_outputs, dim=2)  # Shape: (batch_size, seq_len, output_dim)
        
        output = self.dropout(output)
        return output, gate_logits, topk_indices


class PositionalFFN(nn.Module):
    """
    Positional Feed Forward Network 
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        dropout=0.1
    ):
        super(PositionalFFN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            SwiGLU(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        x
    ):
        return self.dropout(self.ffn(x)), None, None  # Match the MoE output
