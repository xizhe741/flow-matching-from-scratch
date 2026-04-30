"""
Diffusion Transformer (Peebles & Xie 2022) 实现.

来源: 用户在 stable_diffusion/diffusion_model.py 内手写的 DiT, 搬入项目结构.
DiT_block 已抽取到 src/model/modules.py::DiTBlock; 本文件仅保留主类.

接口:
    DiT(forward)(x: (B, in_channels, H, W), t: (B,) | (B,1) | (B,1,1,1))
        -> (B, in_channels, H, W)
    与 U_Net 完全互换.

构造参数:
    Height, Width, patch_size, in_channels  -- 输入图像规格
    hidden_dim, embed_dim                   -- token 隐维 / 时间嵌入维
    num_heads, num_DiTblocks                -- 注意力头数 / 堆叠 block 数
    mlp_ratio                               -- FFN 隐维放大倍率, 默认 4
"""

import torch
from torch import nn

from src.model.modules import DiTBlock, sinusoidal_embedding


class DiT(nn.Module):
    def __init__(
        self,
        Height: int,
        Width: int,
        patch_size: int,
        in_channels: int,
        hidden_dim: int,
        embed_dim: int,
        num_heads: int,
        num_DiTblocks: int,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # patchify + 线性映射到 hidden_dim 一步完成
        self.patchify = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        num_tokens = (Height * Width) // (patch_size ** 2)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))

        self.t_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(num_DiTblocks)]
        )

        # 末端 AdaLN-Zero + linear 映射回像素空间
        self.activate = nn.SiLU()
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_adaLN = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)
        self.final_linear = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 与 U_Net 一致: 接受 (B,) / (B,1) / (B,1,1,1) 形状的 t, 统一压成 (B,)
        if t.dim() > 1:
            t = t.view(-1)
        # t 在 [0, 1] 区间, sinusoidal 高频项几乎归零; 缩放到 [0, 1000] 让全频谱有信号.
        t_scaled = t * 1000.0
        c = self.t_mlp(sinusoidal_embedding(self.embed_dim, t_scaled))

        # (B, C, H, W) -> (B, hidden, H/p, W/p) -> (B, num_tokens, hidden)
        h = self.patchify(x).flatten(2).transpose(1, 2)
        h = h + self.pos_embed

        for block in self.blocks:
            h = block(h, c)

        # 末端 AdaLN-Zero 调制 + 解 patch
        h = self.final_norm(h)
        gamma, beta = self.final_adaLN(self.activate(c)).chunk(2, dim=-1)
        h = (1 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)
        h = self.final_linear(h)

        p = self.patch_size
        num_tokens = h.shape[1]
        side = int(num_tokens ** 0.5)
        assert side * side == num_tokens, (
            f"DiT 假设方形 token 网格, 但 num_tokens={num_tokens} 非完全平方数"
        )
        c_out = self.in_channels
        h = h.reshape(h.shape[0], side, side, p, p, c_out)
        h = h.permute(0, 5, 1, 3, 2, 4)
        h = h.reshape(h.shape[0], c_out, side * p, side * p)
        return h
