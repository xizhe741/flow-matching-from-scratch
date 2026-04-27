"""
U-Net (ResBlock + self-attention + 正弦时间嵌入), 直接复用 diffusion_model_from_scratch.

输入 / 输出形状: (B, 3, 32, 32) 适用于 CIFAR-10.
forward(x, t):
    x: (B, 3, 32, 32)
    t: (B,) 浮点张量, flow matching 在 [0, 1] 上的连续时间.
    返回 (B, 3, 32, 32), 调用方解释为 velocity 或 score.
"""

import torch
import torch.nn as nn

from src.model.modules import (
    Downsample,
    ResBlock,
    Upsample,
    self_attention,
    sinusoidal_embedding,
)


class U_Net(nn.Module):
    def __init__(self, base_channels: int, embedded_dim: int, in_channels: int = 3, **kwargs):
        super().__init__()

        self.entry = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.exit = nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.embedded_dim = embedded_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(embedded_dim, embedded_dim),
            nn.SiLU(),
            nn.Linear(embedded_dim, embedded_dim),
        )

        self.norm = nn.GroupNorm(8, base_channels)
        self.silu = nn.SiLU()

        encoder_channels = [base_channels, base_channels * 2, base_channels * 2, base_channels * 4]
        num_res_blocks = 2
        attn_levels = [False, True, True, False]
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = encoder_channels[0]
        for i, out_ch in enumerate(encoder_channels):
            level = nn.ModuleList()
            for _ in range(num_res_blocks):
                level.append(ResBlock(in_ch, out_ch, embedded_dim))
                in_ch = out_ch
            if attn_levels[i]:
                level.append(self_attention(out_ch))
            self.encoder.append(level)
            if i < len(encoder_channels) - 1:
                self.downsamples.append(Downsample(out_ch))

        self.bottleneck = nn.ModuleList([
            ResBlock(encoder_channels[-1], encoder_channels[-1], embedded_dim),
            self_attention(encoder_channels[-1]),
            ResBlock(encoder_channels[-1], encoder_channels[-1], embedded_dim),
        ])

        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        decoder_channels = encoder_channels[::-1]
        prev_out_ch = encoder_channels[-1]
        for i, out_ch in enumerate(decoder_channels):
            in_ch = prev_out_ch + decoder_channels[i]
            level = nn.ModuleList()
            for _ in range(num_res_blocks):
                level.append(ResBlock(in_ch, out_ch, embedded_dim))
                in_ch = out_ch
            if attn_levels[i]:
                level.append(self_attention(out_ch))
            self.decoder.append(level)
            if i < len(decoder_channels) - 1:
                self.upsamples.append(Upsample(out_ch))
            prev_out_ch = out_ch

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # flow matching 调用时传入的 t 形状可能是 (B,) 或 (B,1)/(B,1,1,1), 这里统一压成 (B,)
        if t.dim() > 1:
            t = t.view(-1)
        t_emb = self.time_mlp(sinusoidal_embedding(self.embedded_dim, t))

        x = self.entry(x)
        skips = []
        for i, level in enumerate(self.encoder):
            for block in level:
                if isinstance(block, ResBlock):
                    x = block(x, t_emb)
                else:
                    x = block(x)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        for block in self.bottleneck:
            if isinstance(block, ResBlock):
                x = block(x, t_emb)
            else:
                x = block(x)

        for i, level in enumerate(self.decoder):
            if i > 0:
                x = self.upsamples[i - 1](x)
            x = torch.cat([x, skips.pop()], dim=1)
            for block in level:
                if isinstance(block, ResBlock):
                    x = block(x, t_emb)
                else:
                    x = block(x)
        x = self.silu(self.norm(x))
        return self.exit(x)
