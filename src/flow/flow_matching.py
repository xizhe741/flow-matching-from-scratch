"""
Stochastic Flow Matching: 联合训练 velocity 网络 v 和 score 网络 s.

训练目标 (eta 参数化, 见 TECHNICAL.md):
    L_v = || v(x_t, t) - xdot_t ||^2,         xdot_t = alpha_dot*x_1 + beta_dot*x_0
    L_s = || s(x_t, t) + z ||^2               (s 学 -z, 即 s_pred = -z_hat)

记号:
    z_hat(x, t)     := E[z | x_t]              (条件噪声后验均值)
    sigma(t)        := 前向插值噪声系数       (Brownian 下 sqrt(2 t (1-t)))
    sigma_dot(t)    := dsigma/dt
    g(t)            := 反向 SDE 扩散系数       (与 sigma 独立的设计自由度)

采样器漂移 (推导见 plan b-a-b-drift-zazzy-avalanche.md 第 5 节):
    z_hat        = -s_pred
    grad log p_t = -z_hat / sigma = s_pred / sigma
    ODE drift    = v + sigma_dot * z_hat       = v - sigma_dot * s_pred
    SDE drift    = v + (sigma_dot - g^2/(2*sigma)) * z_hat
                 = v - (sigma_dot - g^2/(2*sigma)) * s_pred
    SDE 增量     = drift dt + g(t) sqrt(dt) * eps,  eps ~ N(0, I)
"""

import torch

from src.flow.diffusion_coef import ScaledSigma
from src.flow.interpolant import (
    interp,
    linear_coeffs,
    noise_sigma,
    noise_sigmadot,
)


class FlowMatching:
    def __init__(self, interp_func=linear_coeffs, eps_t: float = 1e-2):
        self.interp_func = interp_func
        self.eps_t = eps_t

    def sample_t(self, batch_size: int, device):
        # 用 Beta(2,2) 采样 (避免 t 过于集中在 0/1 端点)
        t = torch.distributions.Beta(2.0, 2.0).sample((batch_size, 1)).to(device)
        return self.eps_t + (1 - 2 * self.eps_t) * t

    def compute_loss(self, net_v, net_s, x_0, x_1):
        """x_0: 噪声样本 (B,C,H,W); x_1: 数据样本."""
        B = x_1.shape[0]
        device = x_1.device
        t = self.sample_t(B, device)
        t_img = t.unsqueeze(-1).unsqueeze(-1)

        x_t, xdot_t, sigma, z = interp(x_0, x_1, t_img, self.interp_func)

        v_pred = net_v(x_t, t)
        s_pred = net_s(x_t, t)

        loss_v = ((v_pred - xdot_t) ** 2).mean()
        # eta 参数化: s_pred 学 -z (DDPM ε-prediction 风格), 端点不爆炸
        loss_s = ((s_pred + z) ** 2).mean()
        return loss_v, loss_s

    def _build_t_grid(self, steps: int, device):
        """在 [eps_t, 1 - eps_t] 上构造长度 steps+1 的均匀时间网格.

        采样器要求时间网格落在 (0, 1) 内部, 否则 sigma_dot 在端点未定义.
        """
        return torch.linspace(self.eps_t, 1.0 - self.eps_t, steps + 1, device=device)

    @torch.no_grad()
    def ode_sample(
        self,
        net_v,
        net_s,
        n: int = None,
        shape=None,
        device=None,
        steps: int = 200,
        x0: torch.Tensor = None,
        t_grid: torch.Tensor = None,
    ):
        """ODE 采样: dx = b_t(x) dt = (v + sigma_dot * z_hat) dt = (v - sigma_dot * s_pred) dt.

        注意 v 不是 ODE 漂移; 必须用完整的 b_t 才能传输与训练 p_t 一致的边际.

        参数:
            net_v, net_s     -- velocity / score 网络
            n, shape, device -- 仅在未传 x0 时生效, 用于初始化 x0 = N(0, I)
            steps            -- 仅在未传 t_grid 时生效, 在 [eps_t, 1-eps_t] 上均匀切分
            x0               -- 显式起点 (退化测试用); 形状 (n, *shape)
            t_grid           -- 显式时间网格 (1D Tensor 或 list); 与 x0 配合使用
        """
        if x0 is None:
            x0 = torch.randn(n, *shape, device=device)
        else:
            n = x0.shape[0]
            device = x0.device

        if t_grid is None:
            t_grid = self._build_t_grid(steps, device)
        else:
            t_grid = torch.as_tensor(t_grid, device=device, dtype=x0.dtype)

        x = x0.clone()
        trajectory = [x.clone()]
        for i in range(len(t_grid) - 1):
            t_val = t_grid[i]
            dt = (t_grid[i + 1] - t_grid[i]).item()
            t_in = torch.full((n, 1), t_val.item(), device=device)

            v = net_v(x, t_in)
            s = net_s(x, t_in)
            sigma_dot = noise_sigmadot(t_val)

            drift = (v - sigma_dot * s) * dt
            x = x + drift
            trajectory.append(x.clone())
        return x, trajectory

    @torch.no_grad()
    def sde_sample(
        self,
        net_v,
        net_s,
        n: int = None,
        shape=None,
        device=None,
        steps: int = 200,
        g_fn=None,
        x0: torch.Tensor = None,
        t_grid: torch.Tensor = None,
    ):
        """反向 SDE 采样 (从 t=eps_t 推到 t=1-eps_t).

        每步 drift = (v - (sigma_dot - g^2/(2 sigma)) * s_pred) dt
              diffusion = g(t) * sqrt(dt) * randn,  最后一步置零

        参数:
            g_fn  -- diffusion_coef.py 里的 ScaledSigma / VPSchedule 等;
                     默认 ScaledSigma(c=1.0), 即 g(t) = sigma(t)
            其余参数同 ode_sample
        """
        if g_fn is None:
            g_fn = ScaledSigma(c=1.0)

        if x0 is None:
            x0 = torch.randn(n, *shape, device=device)
        else:
            n = x0.shape[0]
            device = x0.device

        if t_grid is None:
            t_grid = self._build_t_grid(steps, device)
        else:
            t_grid = torch.as_tensor(t_grid, device=device, dtype=x0.dtype)

        x = x0.clone()
        trajectory = [x.clone()]
        n_steps = len(t_grid) - 1
        for i in range(n_steps):
            t_val = t_grid[i]
            dt = (t_grid[i + 1] - t_grid[i]).item()
            t_in = torch.full((n, 1), t_val.item(), device=device)

            v = net_v(x, t_in)
            s = net_s(x, t_in)
            sigma_t = noise_sigma(t_val)
            sigma_dot_t = noise_sigmadot(t_val)
            g_t = g_fn(t_val)
            g_sq = g_t ** 2

            # s_pred = -z_hat, 故 z_hat 前的系数 (sigma_dot - g^2/(2 sigma)) 取负即作用在 s 上
            coef_on_s = -(sigma_dot_t - g_sq / (2.0 * sigma_t))
            drift = (v + coef_on_s * s) * dt

            if i == n_steps - 1:
                noise = 0.0
            else:
                noise = g_t * (dt ** 0.5) * torch.randn_like(x)

            x = x + drift + noise
            trajectory.append(x.clone())
        return x, trajectory
