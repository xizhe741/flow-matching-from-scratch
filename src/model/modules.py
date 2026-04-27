"""
U-Net 的基础模块, 直接搬自 diffusion_model_from_scratch 项目.
- sinusoidal_embedding: 把标量时间 t 映射到正弦/余弦位置编码
- ResBlock: GroupNorm + SiLU + Conv 的残差块, 在中间注入时间嵌入
- self_attention: 在空间维度上做 self-attention
- Downsample / Upsample: stride=2 卷积下采样 / 最近邻上采样
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_embedding(embedded_dim: int, t: torch.Tensor) -> torch.Tensor:
    """t: (B,) -> (B, embedded_dim)."""
    i = torch.arange(embedded_dim // 2)
    freq = torch.exp((-2 * math.log(10000) * i / embedded_dim)).to(t.device)
    args = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedded_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activate = nn.SiLU()
        self.time_linear = nn.Linear(embedded_dim, out_channels)
        self.embedded_dim = embedded_dim
        self.out_channels = out_channels
        if in_channels == out_channels:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t_emb):
        t_emb = self.time_linear(t_emb).view(-1, self.out_channels, 1, 1)
        res = self.residual_conv(x)
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x) + t_emb
        x = self.norm2(x)
        x = self.activate(x)
        x_res = self.conv2(x) + res
        return x_res


class self_attention(nn.Module):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.toQ = nn.Linear(channels, channels)
        self.toK = nn.Linear(channels, channels)
        self.toV = nn.Linear(channels, channels)
        self.toX = nn.Linear(channels, channels)
        self.norm = nn.GroupNorm(8, channels)
        self.channels = channels

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        x = self.norm(x)
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        Q = self.toQ(x)
        K = self.toK(x)
        V = self.toV(x)
        x = F.scaled_dot_product_attention(Q, K, V)
        x = self.toX(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, -1)
        x_res = x + res
        return x_res


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
