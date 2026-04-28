"""
=============================================================================
 多组采样对比: EMA vs raw, SDE vs ODE
=============================================================================
用同一组初始噪声 x_0 跑 4 种采样方式, 拼成一张大对比图, 直观看哪种最好.

四种组合:
  1) EMA + SDE  (默认, 含 score 修正)
  2) EMA + ODE  (只用 v, 无随机性)
  3) raw + SDE
  4) raw + ODE

输出: 一张 4xN 的网格图, 每行是一种采样方式, 每列是同一个 x_0 出来的样本.

Usage (从项目根目录运行):
    python scripts/compare_sampling.py
    python scripts/compare_sampling.py --n 8 --steps 200 --out compare.png
    python scripts/compare_sampling.py --ckpt checkpoints/epoch_0150.pt
=============================================================================
"""

import argparse
import os
import sys

import torch
from torchvision.utils import make_grid, save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.flow.flow_matching import FlowMatching
from src.flow.interpolant import linear_coeffs
from src.model.U_net import U_Net


def load_weights(net, ckpt, key):
    net.load_state_dict(ckpt[key])
    net.eval()
    return net


@torch.no_grad()
def sample_with_x0(flow, net_v, net_s, x0, steps, use_ode):
    """从给定的 x0 出发采样, 保证不同方式用同一份噪声.

    走 FlowMatching.{ode,sde}_sample 以共享同一套漂移公式
    (含 BFromVS 修正), 不再在脚本内自己拼 drift.
    """
    if use_ode:
        x, _ = flow.ode_sample(net_v, net_s, x0=x0, steps=steps)
    else:
        x, _ = flow.sde_sample(net_v, net_s, x0=x0, steps=steps)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/latest.pt")
    parser.add_argument("--out", default="compare.png")
    parser.add_argument("--n", type=int, default=8, help="每组采样张数")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0, help="x0 随机种子")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    print(f"loaded {args.ckpt}, epoch = {ckpt.get('epoch')}")

    # 同一份初始噪声给所有四种方式用
    torch.manual_seed(args.seed)
    x0 = torch.randn(args.n, 3, 32, 32, device=device)

    flow = FlowMatching(interp_func=linear_coeffs)

    configs = [
        ("EMA + SDE", "ema_v", "ema_s", False),
        ("EMA + ODE", "ema_v", "ema_s", True),
        ("raw + SDE", "net_v", "net_s", False),
        ("raw + ODE", "net_v", "net_s", True),
    ]

    rows = []
    for name, key_v, key_s, use_ode in configs:
        print(f"sampling: {name} ...")
        net_v = U_Net(128, 512).to(device)
        net_s = U_Net(128, 512).to(device)
        load_weights(net_v, ckpt, key_v)
        load_weights(net_s, ckpt, key_s)
        x = sample_with_x0(flow, net_v, net_s, x0, args.steps, use_ode)
        x = (x.clamp(-1, 1) + 1) / 2
        rows.append(x)
        del net_v, net_s
        torch.cuda.empty_cache()

    # 拼接: 4 行 x n 列
    grid_input = torch.cat(rows, dim=0)        # (4*n, 3, 32, 32)
    grid = make_grid(grid_input, nrow=args.n)  # 每行 n 张
    save_image(grid, args.out)
    print(f"\nsaved -> {os.path.abspath(args.out)}")
    print(f"行从上到下依次是: {' / '.join(c[0] for c in configs)}")


if __name__ == "__main__":
    main()
