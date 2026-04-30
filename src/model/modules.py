"""
模型基础模块.

U-Net 子模块 (来源: diffusion_model_from_scratch):
- sinusoidal_embedding: 标量时间 t -> 正弦/余弦位置编码
- ResBlock:             GroupNorm + SiLU + Conv 残差块, 中间注入时间嵌入
- self_attention:       空间维度 self-attention
- Downsample / Upsample: stride=2 卷积下采样 / 最近邻上采样

DiT 子模块 (来源: stable_diffusion/diffusion_model.py):
- DiTBlock:             AdaLN-Zero 条件化的 Transformer block (Peebles & Xie 2022)
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


class DiTBlock(nn.Module):
    """AdaLN-Zero Transformer block (Peebles & Xie 2022, "Scalable Diffusion Models with Transformers").

    Forward:
        x: (B, num_tokens, hidden_dim)
        c: (B, hidden_dim)  条件嵌入 (本项目里是时间嵌入)
    输出与 x 同形状.

    每个 block 由条件 c 预测六个 AdaLN 调制参数
        gamma1, beta1, alpha1, gamma2, beta2, alpha2,
    分别用于 attention 与 FFN 两支的 LayerNorm 调制和残差缩放.
    `Linear(c -> 6 * hidden_dim)` 的权重/偏置零初始化, 保证训练初期 block 等价于恒等映射 (Zero 初始化的 AdaLN).
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.activate = nn.SiLU()
        self.adaLN_modulation = nn.Linear(hidden_dim, hidden_dim * 6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_ratio * hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_ratio * hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = (
            self.adaLN_modulation(self.activate(c)).chunk(6, dim=-1)
        )

        # attention branch
        h1 = (1 + gamma1.unsqueeze(1)) * self.norm_attn(x) + beta1.unsqueeze(1)
        x = x + alpha1.unsqueeze(1) * self.attention(h1, h1, h1)[0]

        # ffn branch
        h2 = (1 + gamma2.unsqueeze(1)) * self.norm_ffn(x) + beta2.unsqueeze(1)
        x = x + alpha2.unsqueeze(1) * self.ffn(h2)
        return x
