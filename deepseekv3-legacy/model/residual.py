import torch
import torch.nn as nn


class AddNorm(nn.Module):
    """
    Residual connection followed by layer normalization.
    
    This module adds the input tensor to the output of a sub-layer (residual connection)
    and then applies layer normalization to the result.
    """
    def __init__(
        self,
        normalized_shape,
        eps=1e-6
    ):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
    
    def forward(
        self,
        x,
        sublayer_output
    ):
        # Residual connection
        added = x + sublayer_output
        # Layer normalization
        normalized = self.layer_norm(added)
        return normalized
