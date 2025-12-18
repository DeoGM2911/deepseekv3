import torch
from torch import nn


class PosEncoding(nn.Module):
    """Absolute positional encoding as in the standard transformer architecture"""
    def __init__(
        self, 
        max_len, 
        input_dim
    ):
        super(PosEncoding, self).__init__()
        self.max_len = max_len
        self.input_dim = input_dim
        self.pos_enc = torch.zeros(max_len, input_dim)
        self.pos_enc[..., ::2] = torch.sin(torch.arange(max_len).unsqueeze(-1) / (10000 ** (2 * torch.arange(0, input_dim, 2) / input_dim)))
        self.pos_enc[..., 1::2] = torch.cos(torch.arange(max_len).unsqueeze(-1) / (10000 ** (2 * torch.arange(1, input_dim, 2) / input_dim)))

    def forward(
        self, 
        x
    ):
        return x + self.pos_enc[:x.size(1), :].unsqueeze(0)


def pre_compute_theta(
    max_len,
    rope_dim
):
    theta = 10000 ** (-2 * torch.arange(0, rope_dim, 2) / rope_dim)
    m_theta = torch.arange(max_len).unsqueeze(-1) * theta  # (seq_len, rope_dim/2)
    m_theta = m_theta.view(1, max_len, 1, -1)
    return m_theta


def rotary_emb(
    x,
    m_theta
):
    """
    Apply rotary position embeddings to the input x.
    """
    # x: (batch, seq_len, num_heads, head_dim)
    # Split into pairs of dimensions
    x1, x2 = x[..., ::2], x[..., 1::2]  # Even and odd dimensions
    
    # Only use the m_theta up to the sequence length
    m_theta = torch.split(m_theta, x.size(1), dim=1)[0]
    
    # Apply rotation
    cos_m = torch.cos(m_theta)
    sin_m = torch.sin(m_theta)
    
    # Rotate pairs
    x_rotated = torch.cat([
        x1 * cos_m - x2 * sin_m,  # Real part
        x1 * sin_m + x2 * cos_m   # Imaginary part
    ], dim=-1)
    
    return x_rotated
