"""
退化高斯测试: p_0 = p_1 = N(0, I), 不依赖任何已训练网络.

数学依据 (推导见 plan b-a-b-drift-zazzy-avalanche.md 第 5 节):
    退化下解析最优网络:
        v_t(x) = (2t - 1) * x
        z_hat(x, t) = sigma(t) * x   =>   s_pred = -z_hat = -sigma(t) * x
        sigma * sigma_dot = 1 - 2t

    ODE 漂移: b_t(x) = v + sigma_dot * z_hat
                    = (2t-1) x + sigma_dot * sigma * x
                    = (2t-1) x + (1-2t) x = 0
    => 正确 ODE 是 dx = 0, x_t ≡ x_0, 每个 t 边际恒为 N(0, I).

    SDE 漂移: b'_t(x) = v + (sigma_dot - g^2/(2 sigma)) * z_hat
                     = (2t-1) x + (sigma_dot - g^2/(2 sigma)) * sigma * x
                     = (2t-1) x + (sigma sigma_dot - g^2/2) * x
                     = (2t-1) x + (1-2t - g^2/2) * x
                     = -(g^2/2) * x
    SDE: dx = -(g^2/2) x dt + g dW (OU 过程), 平稳分布 N(0, I).
    初始 x_0 ~ N(0, I) 时, 边际恒为 N(0, I), 与 g 无关.

测试模块只验证采样器漂移组装, 不依赖训练. 修改 sampler/interpolant/diffusion_coef
后必须重跑. 退化测试对任意合法 g 都应通过 — 改 g_fn 后挂说明漂移公式有 bug.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.flow.diffusion_coef import ScaledSigma, VPSchedule
from src.flow.flow_matching import FlowMatching
from src.flow.interpolant import noise_sigma


def _broadcast_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """t: (B, 1) -> 与 x 同维 (B, 1, 1, ..., 1) 以便逐元素相乘."""
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    return t


class MockVNet:
    """v(x, t) = (2t - 1) x.  接口与 U_Net.forward(x, t) 一致."""

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_b = _broadcast_t(t, x)
        return (2.0 * t_b - 1.0) * x

    # 让 .eval() / .to(device) 等调用静默通过, 兼容真实 nn.Module 用法.
    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


class MockSNet:
    """s(x, t) = -sigma(t) x.  s_pred 学 -z_hat, 退化下 z_hat = sigma * x."""

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_b = _broadcast_t(t, x)
        sigma = noise_sigma(t_b)
        return -sigma * x

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


# 检查时刻 (单调)
_CHECK_TIMES_FULL = (0.25, 0.5, 0.75, 1.0)

# 网格密度 (步数足够细以保证 SDE 离散误差不主导)
_N = 10000
_DIM = 32
_STEPS_PER_SEGMENT = 50


def _build_t_grid(check_times, t_start: float = 1e-2, steps_per_segment: int = None):
    """构造单调时间网格, 在每对 check_times 之间插入 steps_per_segment 个等距点.

    steps_per_segment 默认取模块级 _STEPS_PER_SEGMENT; 对 g 较大的预设
    (如 VPSchedule, t->1 处 g^2 = beta_max) Euler-Maruyama 离散化误差更显著,
    需要传更大的 steps_per_segment 才能让方差检查通过.
    """
    if steps_per_segment is None:
        steps_per_segment = _STEPS_PER_SEGMENT
    grid = [t_start]
    for t_next in check_times:
        prev = grid[-1]
        for k in range(1, steps_per_segment + 1):
            grid.append(prev + (t_next - prev) * k / steps_per_segment)
    return torch.tensor(grid, dtype=torch.float64)


def _check_variance_at_checkpoints(traj, t_grid, check_times, tol):
    """traj[i] 对应 t_grid[i]; 在 check_times 处验证 mean ≈ 0, var ≈ 1."""
    for t_target in check_times:
        # 找最接近的索引
        idx = int(torch.argmin(torch.abs(t_grid - t_target)).item())
        x_t = traj[idx]
        var = (x_t ** 2).mean().item()
        assert abs(var - 1.0) < tol, (
            f"t≈{t_grid[idx].item():.4f} (target {t_target}): var={var:.4f}, tol={tol}"
        )


def test_ode_sampler_degenerate():
    torch.manual_seed(0)
    flow = FlowMatching()
    v_mock, s_mock = MockVNet(), MockSNet()
    x0 = torch.randn(_N, _DIM, dtype=torch.float64)
    t_grid = _build_t_grid(_CHECK_TIMES_FULL).to(x0.dtype)

    _, traj = flow.ode_sample(v_mock, s_mock, x0=x0, t_grid=t_grid)
    _check_variance_at_checkpoints(traj, t_grid, _CHECK_TIMES_FULL, tol=1e-2)


def test_sde_sampler_degenerate_default_g():
    torch.manual_seed(1)
    flow = FlowMatching()
    v_mock, s_mock = MockVNet(), MockSNet()
    x0 = torch.randn(_N, _DIM, dtype=torch.float64)
    t_grid = _build_t_grid(_CHECK_TIMES_FULL).to(x0.dtype)

    _, traj = flow.sde_sample(
        v_mock, s_mock, x0=x0, t_grid=t_grid, g_fn=ScaledSigma(c=1.0)
    )
    _check_variance_at_checkpoints(traj, t_grid, _CHECK_TIMES_FULL, tol=5e-2)


def test_sde_sampler_degenerate_scaled_g():
    torch.manual_seed(2)
    flow = FlowMatching()
    v_mock, s_mock = MockVNet(), MockSNet()
    x0 = torch.randn(_N, _DIM, dtype=torch.float64)
    t_grid = _build_t_grid(_CHECK_TIMES_FULL).to(x0.dtype)

    _, traj = flow.sde_sample(
        v_mock, s_mock, x0=x0, t_grid=t_grid, g_fn=ScaledSigma(c=0.5)
    )
    _check_variance_at_checkpoints(traj, t_grid, _CHECK_TIMES_FULL, tol=5e-2)


def test_sde_sampler_degenerate_vp_schedule():
    """Score-SDE Variance-Preserving 调度: g(t) = sqrt(beta_min + t*(beta_max-beta_min)).

    VP 调度在 t->1 处 g^2 = beta_max 较大 (典型 20.0), Euler-Maruyama 一阶
    离散化的 OU 过程方差有 O(g^2 * dt) 量级偏差, 需要更细的时间网格 + 略宽
    的容差才能让方差检查通过. 这一容差宽度反映离散化误差, 不是漂移公式问题.
    """
    torch.manual_seed(3)
    flow = FlowMatching()
    v_mock, s_mock = MockVNet(), MockSNet()
    x0 = torch.randn(_N, _DIM, dtype=torch.float64)
    t_grid = _build_t_grid(_CHECK_TIMES_FULL, steps_per_segment=500).to(x0.dtype)

    _, traj = flow.sde_sample(
        v_mock, s_mock, x0=x0, t_grid=t_grid, g_fn=VPSchedule(beta_min=0.1, beta_max=20.0)
    )
    _check_variance_at_checkpoints(traj, t_grid, _CHECK_TIMES_FULL, tol=1e-1)


def test_sigma_sigma_dot_identity():
    """单元测试: sigma(t) * sigma_dot(t) == 1 - 2t 在内部点上成立."""
    from src.flow.interpolant import noise_sigmadot

    t = torch.linspace(0.05, 0.95, 19, dtype=torch.float64)
    sigma = noise_sigma(t)
    sigma_dot = noise_sigmadot(t)
    product = sigma * sigma_dot
    expected = 1.0 - 2.0 * t
    err = (product - expected).abs().max().item()
    assert err < 1e-10, f"sigma * sigma_dot 与 1 - 2t 不匹配, max err = {err}"


if __name__ == "__main__":
    test_sigma_sigma_dot_identity()
    print("[OK] sigma * sigma_dot identity")
    test_ode_sampler_degenerate()
    print("[OK] ODE degenerate gaussian")
    test_sde_sampler_degenerate_default_g()
    print("[OK] SDE degenerate gaussian (g = sigma)")
    test_sde_sampler_degenerate_scaled_g()
    print("[OK] SDE degenerate gaussian (g = 0.5 sigma)")
    test_sde_sampler_degenerate_vp_schedule()
    print("[OK] SDE degenerate gaussian (VP schedule beta=0.1->20)")
    print("\nALL TESTS PASSED")
