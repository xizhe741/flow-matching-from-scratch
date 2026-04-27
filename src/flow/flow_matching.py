"""
Stochastic Flow Matching: 联合训练 velocity field v 和 score field s.

训练损失:
    L_v = || v(x_t, t) - xdot_t ||^2
    L_s = || sigma(t) * s(x_t, t) + z ||^2     (denoising score matching)

反向 SDE 采样:
    dx = (v + 0.5 * sigma(t)^2 * s) dt + sigma(t) dW
"""

import torch

from src.flow.interpolant import interp, linear_coeffs, noise_magnify


class FlowMatching:
    def __init__(self, interp_func=linear_coeffs, sigma_func=noise_magnify, eps_t: float = 1e-2):
        self.interp_func = interp_func
        self.sigma_func = sigma_func
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

        x_t, xdot_t, sigma, z = interp(x_0, x_1, t_img, self.interp_func, self.sigma_func)

        v_pred = net_v(x_t, t)
        s_pred = net_s(x_t, t)

        loss_v = ((v_pred - xdot_t) ** 2).mean()
        loss_s = ((sigma * s_pred + z) ** 2).mean()
        return loss_v, loss_s

    @torch.no_grad()
    def ode_sample(self, net_v, n: int, shape, device, steps: int = 200):
        """纯 ODE 采样: dx = v(x, t) dt. 不需要 score 网络, 也没有随机性.
        如果 v 学好了, 这条路径应该比 SDE 更稳, 是诊断模型质量的好工具.
        """
        x = torch.randn(n, *shape, device=device)
        dt = 1.0 / steps
        trajectory = [x.clone()]
        for i in range(steps):
            t_val = i * dt
            t_current = torch.full((n, 1), t_val, device=device)
            v = net_v(x, t_current)
            x = x + v * dt
            trajectory.append(x.clone())
        return x, trajectory

    @torch.no_grad()
    def sde_sample(self, net_v, net_s, n: int, shape, device, steps: int = 200):
        """从 x_0 ~ N(0,I) 出发, 用反向 SDE 推到 x_1.

        返回:
            x          -- 最终生成的样本 (n, *shape)
            trajectory -- 长度 steps+1 的 list, 记录每一步的 x.
                          trajectory[0] 是初始噪声, trajectory[steps] 是最终结果.
                          可视化脚本可以从中挑任意时刻 (例如 t=0, 0.25, 0.5, 0.75, 1.0) 画出中间过程.
                          如果只需要最终图像可以忽略它; 注意每步会 clone 一次, 步数大时会占显存.
        """
        x = torch.randn(n, *shape, device=device)
        dt = 1.0 / steps
        trajectory = [x.clone()]
        for i in range(steps):
            t_val = i * dt
            t_current = torch.full((n, 1), t_val, device=device)
            v = net_v(x, t_current)
            s = net_s(x, t_current)
            sigma_t, _ = self.sigma_func(torch.tensor(t_val, device=device))

            drift = (v + 0.5 * sigma_t ** 2 * s) * dt
            if i == steps - 1:
                diffusion = 0.0
            else:
                diffusion = sigma_t * (dt ** 0.5) * torch.randn_like(x)

            x = x + drift + diffusion
            trajectory.append(x.clone())
        return x, trajectory
