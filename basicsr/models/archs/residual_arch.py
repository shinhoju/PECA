import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5):
        super(ResidualConnection, self).__init__()
        self.norm=nn.LayerNorm(d_model, eps=eps)
        self.norm2=nn.LayerNorm(d_model, eps=eps)
        self.sublayer = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.output = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        x2 = x2.permute(0, 2, 3, 1)                     # (B, H, W, C)
        x2_norm = self.norm(x2).permute(0, 3, 1, 2)     # (B, C, H, W)
        y = self.sublayer(x2_norm)
        
        y = x1 + y
        y = y.permute(0, 2, 3, 1)
        out = self.norm2(y).permute(0, 3, 1, 2)
        out = self.output(out)

        return y.permute(0, 3, 1, 2) + out
