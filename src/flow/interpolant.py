"""
插值方案 (interpolant) 与噪声放大函数 (sigma).

x_t = alpha(t) * x_1 + beta(t) * x_0 + sigma(t) * z,   z ~ N(0, I)

提供:
1. noise_sigma     -- sigma(t) = sqrt(2 t (1-t))
2. noise_sigmadot  -- sigma_dot(t) = (1 - 2t) / sigma(t),  满足 sigma * sigma_dot = 1 - 2t
3. linear_coeffs   -- alpha = t,         beta = 1 - t
4. trig_coeffs     -- alpha = sin(pi t/2), beta = cos(pi t/2)
5. interp          -- 给定 x0, x1, t, 返回 x_t / xdot_t / sigma / z
"""

from typing import Tuple
import numpy as np
import torch


def noise_sigma(t: torch.Tensor) -> torch.Tensor:
    """sigma(t) = sqrt(2 t (1-t)).  端点处为 0."""
    sigma_sq = 2.0 * t * (1.0 - t)
    sigma_sq = torch.clamp(sigma_sq, min=0.0)
    return torch.sqrt(sigma_sq)


def noise_sigmadot(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """sigma_dot(t) = (1 - 2t) / sigma(t).

    解析关系 sigma * sigma_dot = 1 - 2t. 端点处理:
        t = 0:  sigma_dot 解析上 -> +inf, 这里 mask 为 0 (避免 NaN/inf 进入采样器)
        t = 1:  sigma_dot 解析上 -> -inf, 同上 mask 为 0
    项目采样网格在 (0,1) 内部时端点 mask 不会触发, 数值与解析一致.
    """
    sigma_sq = 2.0 * t * (1.0 - t)
    sigma_dot = (1.0 - 2.0 * t) / torch.sqrt(torch.clamp(sigma_sq, min=eps))
    endpoint_mask = (t <= 0.0) | (t >= 1.0)
    return torch.where(endpoint_mask, torch.zeros_like(sigma_dot), sigma_dot)


# TODO(decouple): σ, σ̇ 应从返回元组中移除, caller 直接调用 noise_sigma / noise_sigmadot.
# 当前为减少本次 PR 影响面暂保留耦合返回。
def linear_coeffs(t: torch.Tensor):
    sigma = noise_sigma(t)
    sigma_dot = noise_sigmadot(t)
    return t, 1 - t, torch.ones_like(t), -torch.ones_like(t), sigma, sigma_dot


# TODO(decouple): σ, σ̇ 应从返回元组中移除, caller 直接调用 noise_sigma / noise_sigmadot.
# 当前为减少本次 PR 影响面暂保留耦合返回。
def trig_coeffs(t: torch.Tensor):
    sigma = noise_sigma(t)
    sigma_dot = noise_sigmadot(t)
    alpha = torch.sin(0.5 * np.pi * t)
    beta = torch.cos(0.5 * np.pi * t)
    alpha_dot = 0.5 * np.pi * torch.cos(0.5 * np.pi * t)
    beta_dot = -0.5 * np.pi * torch.sin(0.5 * np.pi * t)
    return alpha, beta, alpha_dot, beta_dot, sigma, sigma_dot


def interp(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    interp_func=linear_coeffs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """t 形状需要广播到与 x 兼容, 例如 (B, 1, 1, 1).

    返回:
        x_t    -- 带噪声插值, 网络的输入
        xdot_t -- v 的训练 target, 不含 sigma_dot * z 噪声项
                  (理论上 E[sigma_dot * z | x_t] = 0, 加上只会增大方差,
                   导致训练单 batch loss 噪声大)
        sigma  -- 当前 t 下的 sigma(t), 用于 score loss 和 SDE 采样
        z      -- 注入的噪声, 用作 score 网络的训练 target
    """
    alpha, beta, alpha_dot, beta_dot, sigma, _sigma_dot = interp_func(t)
    z = torch.randn_like(x_1)
    x_t = alpha * x_1 + beta * x_0 + sigma * z
    # v 学 dx_t/dt 的期望部分, sigma_dot * z 的期望为 0, 不参与训练 target
    xdot_t = alpha_dot * x_1 + beta_dot * x_0
    return x_t, xdot_t, sigma, z
