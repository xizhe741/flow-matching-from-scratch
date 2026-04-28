"""
反向 SDE 采样的扩散系数 g(t).

g(t) 与训练时插值用的 sigma(t) 是**独立的设计自由度**:
    - sigma(t): 决定前向插值轨迹 x_t = alpha*x_1 + beta*x_0 + sigma(t)*z 的展宽,
      训练时固定为 sqrt(2 t (1-t)) (Brownian 选择).
    - g(t):    决定反向 SDE 噪声幅度 dx = drift dt + g(t) dW,
      论文明确称之为 "tunable diffusion coefficient".

接口约定: g_fn(t: Tensor) -> Tensor, 输入时间张量, 返回相同形状的非负张量.
采样器入口处应保证 g(t)/sigma(t) 在网格上有限 (端点 sigma -> 0 时 g 必须同速或更快地趋零).

提供两个预设:
    ScaledSigma(c)    -- g(t) = c * sigma(t).  c=0 退化为 ODE 极限, c=1 为默认.
    SigmaSigmaDot()   -- g(t)^2 = 2 * sigma * sigma_dot = 2 (1-2t) (Brownian).
                         注意 t > 1/2 时 sigma*sigma_dot < 0, g 无定义,
                         调用时若 t > 0.5 抛出异常.
"""

import torch

from src.flow.interpolant import noise_sigma, noise_sigmadot


class ScaledSigma:
    """g(t) = c * sigma(t).  c >= 0 标量超参."""

    def __init__(self, c: float = 1.0):
        if c < 0:
            raise ValueError(f"ScaledSigma: c must be >= 0, got {c}")
        self.c = c

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.c * noise_sigma(t)


class SigmaSigmaDot:
    """g(t)^2 = 2 * sigma(t) * sigma_dot(t) = 2 (1-2t) (Brownian 选择).

    仅在 t <= 0.5 区间合法; t > 0.5 时 sigma*sigma_dot < 0, 平方为负, g 无定义.
    对应 score-SDE 文献中某种 VP 形式, 仅适用于前半段时间.
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if torch.any(t > 0.5):
            raise ValueError(
                "SigmaSigmaDot: g^2 = 2*sigma*sigma_dot < 0 when t > 0.5; "
                "this g choice is only valid on t in [0, 0.5]."
            )
        g_sq = 2.0 * noise_sigma(t) * noise_sigmadot(t)
        g_sq = torch.clamp(g_sq, min=0.0)
        return torch.sqrt(g_sq)
