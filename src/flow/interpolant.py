"""
插值方案 (interpolant) 与噪声放大函数 (sigma).

x_t = alpha(t) * x_1 + beta(t) * x_0 + sigma(t) * z,   z ~ N(0, I)

提供:
1. noise_magnify  -- sigma(t) = sqrt(2 t (1-t))
2. linear_coeffs  -- alpha = t,         beta = 1 - t
3. trig_coeffs    -- alpha = sin(pi t/2), beta = cos(pi t/2)
4. interp         -- 给定 x0, x1, t, 返回 x_t / xdot_t / sigma / z
"""

from typing import Tuple
import numpy as np
import torch


def noise_magnify(t: torch.Tensor, eps: float = 1e-12):
    sigma_sq = 2.0 * t * (1.0 - t)
    sigma_sq = torch.clamp(sigma_sq, min=0.0)
    sigma = torch.sqrt(sigma_sq)
    sigma_dot = (1.0 - 2.0 * t) / torch.sqrt(torch.clamp(sigma_sq, min=eps))

    endpoint_mask = (t <= 0.0) | (t >= 1.0)
    sigma = torch.where(endpoint_mask, torch.zeros_like(sigma), sigma)
    sigma_dot = torch.where(endpoint_mask, torch.zeros_like(sigma_dot), sigma_dot)
    return sigma, sigma_dot


def linear_coeffs(t: torch.Tensor, sigma_func=noise_magnify):
    sigma, sigma_dot = sigma_func(t)
    return t, 1 - t, torch.ones_like(t), -torch.ones_like(t), sigma, sigma_dot


def trig_coeffs(t: torch.Tensor, sigma_func=noise_magnify):
    sigma, sigma_dot = sigma_func(t)
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
    sigma_func=noise_magnify,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """t 形状需要广播到与 x 兼容, 例如 (B, 1, 1, 1)."""
    alpha, beta, alpha_dot, beta_dot, sigma, sigma_dot = interp_func(t, sigma_func)
    z = torch.randn_like(x_1)
    x_t = alpha * x_1 + beta * x_0 + sigma * z
    xdot_t = alpha_dot * x_1 + beta_dot * x_0 + sigma_dot * z
    return x_t, xdot_t, sigma, z
