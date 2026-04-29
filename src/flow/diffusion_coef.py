"""
反向 SDE 采样的扩散系数 g(t).

g(t) 与训练时插值用的 sigma(t) 是独立的设计自由度:
    - sigma(t): 决定前向插值轨迹 x_t = alpha*x_1 + beta*x_0 + sigma(t)*z 的展宽,
      训练时固定为 sqrt(2 t (1-t)) (Brownian 选择).
    - g(t):    决定反向 SDE 噪声幅度 dx = drift dt + g(t) dW,
      论文 (Albergo et al. 2023) 中称之为 tunable diffusion coefficient.

接口约定: g_fn(t: Tensor) -> Tensor, 输入时间张量, 返回相同形状的非负张量.
采样器入口处应保证 g(t)/sigma(t) 在网格上有限 (端点 sigma -> 0 时 g 必须同速或更快地趋零;
若使用与 sigma 解耦的 g 形式, 必须依赖 t_grid 截断 [eps_t, 1-eps_t] 缓解).

提供两个预设:
    ScaledSigma(c)              -- g(t) = c * sigma(t).  c=0 退化为 ODE 极限, c=1 为默认.
    VPSchedule(beta_min, beta_max)
                                -- Score-SDE (Song et al. 2021) Variance-Preserving 形式:
                                   beta(t) = beta_min + t * (beta_max - beta_min)
                                   g(t)   = sqrt(beta(t))
                                   beta_min/beta_max 默认 (0.1, 20.0), 与原文一致.
                                   注意: 该调度与本项目前向插值 sigma 解耦,
                                   仅作为反向 SDE 的扩散系数旋钮提供.
"""

import torch

from src.flow.interpolant import noise_sigma


class ScaledSigma:
    """g(t) = c * sigma(t).  c >= 0 标量超参."""

    def __init__(self, c: float = 1.0):
        if c < 0:
            raise ValueError(f"ScaledSigma: c must be >= 0, got {c}")
        self.c = c

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.c * noise_sigma(t)


class VPSchedule:
    """Variance-Preserving 反向 SDE 噪声调度.

    Song et al. ICLR 2021, "Score-Based Generative Modeling through SDEs":
        前向 VP-SDE:  dx = -0.5 beta(t) x dt + sqrt(beta(t)) dW
    本项目复用其 g(t) = sqrt(beta(t)) 作为反向 SDE 的扩散系数,
    beta(t) = beta_min + t * (beta_max - beta_min).

    与 ScaledSigma 不同, VP 调度不依赖训练时的 sigma(t); 端点附近 g(t) 不归零,
    需配合采样器的 t_grid 端点截断使用.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        if beta_min < 0 or beta_max < 0:
            raise ValueError(
                f"VPSchedule: beta_min and beta_max must be >= 0, "
                f"got beta_min={beta_min}, beta_max={beta_max}"
            )
        if beta_max < beta_min:
            raise ValueError(
                f"VPSchedule: beta_max must be >= beta_min, "
                f"got beta_min={beta_min}, beta_max={beta_max}"
            )
        self.beta_min = beta_min
        self.beta_max = beta_max

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        return torch.sqrt(beta_t)
