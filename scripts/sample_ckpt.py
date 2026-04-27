"""
=============================================================================
 ⚠️ 重点阅读: 这个文件是 flow matching 的反向 SDE 采样入口
=============================================================================

它做的事和原 flow_matching.py 第 226-269 行**完全一样**:
从纯噪声 x_0 ~ N(0, I) 出发, 按反向 SDE 走 steps 步推到 x_1.

核心 SDE (在 src/flow/flow_matching.py 的 sde_sample 里实现):
    drift     = (v(x_t, t) + 0.5 * sigma(t)^2 * s(x_t, t)) * dt
    diffusion = sigma(t) * sqrt(dt) * z,    z ~ N(0, I)
    x_{t+dt}  = x_t + drift + diffusion       # 最后一步令 diffusion = 0

和原代码的对应关系:
    sampling_p0(16)        -> sde_sample 内部 torch.randn(n, *shape, ...)
    for i in range(steps): -> sde_sample 的 for 循环
    vector_net_v / _s      -> 这里加载 checkpoint 后的 net_v / net_s
    plt.imshow + 子图       -> torchvision.utils.save_image (8x8 网格 PNG)

与原代码相比的两点不同:
1. 从硬盘读 checkpoint, 这样训练和采样解耦, 训练完后想采几次都行
2. 默认使用 EMA 权重 (ema_v / ema_s); EMA 通常比 raw 权重生成更稳定

Usage (从项目根目录运行):
    python scripts/sample_ckpt.py
    python scripts/sample_ckpt.py --ckpt checkpoints/latest.pt --n 64 --out samples.png
    python scripts/sample_ckpt.py --use-model   # 用原模型权重而不是 EMA
=============================================================================
"""

import argparse
import os
import sys

import torch
from torchvision.utils import make_grid, save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.flow.flow_matching import FlowMatching
from src.flow.interpolant import linear_coeffs, noise_magnify
from src.model.U_net import U_Net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/latest.pt")
    parser.add_argument("--out", default="samples.png")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--use-model", action="store_true",
                        help="Use raw model weights instead of EMA (default: EMA)")
    parser.add_argument("--ode", action="store_true",
                        help="Use pure ODE sampling (only v, no score, deterministic)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    print(f"loaded {args.ckpt}, keys = {list(ckpt.keys())}, epoch = {ckpt.get('epoch')}")

    net_v = U_Net(128, 512).to(device)
    net_s = U_Net(128, 512).to(device)
    if args.use_model:
        net_v.load_state_dict(ckpt["net_v"])
        net_s.load_state_dict(ckpt["net_s"])
        print("using raw model weights")
    else:
        net_v.load_state_dict(ckpt["ema_v"])
        net_s.load_state_dict(ckpt["ema_s"])
        print("using EMA weights")
    net_v.eval()
    net_s.eval()

    flow = FlowMatching(interp_func=linear_coeffs, sigma_func=noise_magnify)

    # ---- 反向采样 (核心): SDE 见 FlowMatching.sde_sample, ODE 见 ode_sample ----
    if args.ode:
        print(f"sampling {args.n} images over {args.steps} ODE steps (v only)...")
        x, _trajectory = flow.ode_sample(
            net_v, n=args.n, shape=(3, 32, 32), device=device, steps=args.steps,
        )
    else:
        print(f"sampling {args.n} images over {args.steps} SDE steps (v + s)...")
        x, _trajectory = flow.sde_sample(
            net_v, net_s, n=args.n, shape=(3, 32, 32), device=device, steps=args.steps,
        )

    # 训练时数据被归一化到 [-1, 1], 这里拉回 [0, 1] 才能正常显示
    x = (x.clamp(-1, 1) + 1) / 2
    grid = make_grid(x, nrow=8)
    save_image(grid, args.out)
    print(f"saved -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
