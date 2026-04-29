"""
=============================================================================
 g(t) 扩散系数 sweep: 一张大图对比 6 种采样预设
=============================================================================
布局: 6 行 x N 列, 同一份 x0 喂给所有预设, 保证横向可比.

行         标签                            sampler        g(t)
1          ODE                             ode_sample     -
2          SDE g=0.5 sigma                 sde_sample     ScaledSigma(0.5)
3          SDE g=1.0 sigma                 sde_sample     ScaledSigma(1.0)  (默认)
4          SDE g=2.0 sigma                 sde_sample     ScaledSigma(2.0)
5          SDE g=3.0 sigma                 sde_sample     ScaledSigma(3.0)
6          SDE VP(beta=0.1->20)            sde_sample     VPSchedule(0.1, 20)

注意:
- VP 行使用 Score-SDE (Song et al. 2021) 形式 g(t) = sqrt(beta(t)),
  beta(t) = beta_min + t*(beta_max - beta_min). 与训练时 sigma(t) 解耦,
  端点不归零, 完全依赖 sampler 内部 t_grid 的 [eps_t, 1-eps_t] 截断.
- c 越大端点附近修正越剧烈, 大 c 建议同时调大 --steps.
- 所有行共享同一份 x0; SDE 行内部仍各自消耗布朗增量随机数,
  故 SDE 行之间的差异既来自 g 也来自 dW 实现.

Usage:
    python scripts/sweep_g.py --ckpt checkpoints/latest.pt --n 24 --out sweep_g.png
=============================================================================
"""

import argparse
import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid, save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.flow.diffusion_coef import ScaledSigma, VPSchedule
from src.flow.flow_matching import FlowMatching
from src.flow.interpolant import linear_coeffs
from src.model.U_net import U_Net


def build_presets():
    """返回 [(label, kind, g_fn_or_none)].

    kind:
        "ode"  -- 全程 ODE
        "sde"  -- 全程 SDE, 用给定 g_fn
    """
    return [
        ("ODE",                          "ode", None),
        ("SDE g=0.5 sigma",              "sde", ScaledSigma(0.5)),
        ("SDE g=1.0 sigma",              "sde", ScaledSigma(1.0)),
        ("SDE g=2.0 sigma",              "sde", ScaledSigma(2.0)),
        ("SDE g=3.0 sigma",              "sde", ScaledSigma(3.0)),
        ("SDE VP (beta 0.1 -> 20)",      "sde", VPSchedule(0.1, 20.0)),
    ]


@torch.no_grad()
def run_preset(flow, net_v, net_s, x0, steps, kind, g_fn):
    if kind == "ode":
        x, _ = flow.ode_sample(net_v, net_s, x0=x0, steps=steps)
        return x

    if kind == "sde":
        x, _ = flow.sde_sample(net_v, net_s, x0=x0, steps=steps, g_fn=g_fn)
        return x

    raise ValueError(f"unknown preset kind: {kind}")


def add_row_labels(grid_path, labels, img_h=32, padding=2, label_width=320, font_size=16):
    """在拼好的 grid 图左侧贴每行标签.

    make_grid(padding=p) 在每张图四周都加 p 像素, 相邻图共享 padding.
    最终第 i 行 (0-indexed) 的图像中心纵坐标 = padding + i*(img_h + padding) + img_h/2.
    """
    img = Image.open(grid_path).convert("RGB")
    W, H = img.size
    canvas = Image.new("RGB", (W + label_width, H), color=(255, 255, 255))
    canvas.paste(img, (label_width, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    stride = img_h + padding
    for i, label in enumerate(labels):
        y_center = padding + i * stride + img_h // 2
        draw.text((10, y_center - font_size // 2), label, fill=(0, 0, 0), font=font)

    canvas.save(grid_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/latest.pt")
    parser.add_argument("--out", default="sweep_g.png")
    parser.add_argument("--n", type=int, default=24, help="每行采样张数")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-raw", action="store_true",
                        help="用原网络权重而非 EMA (默认 EMA)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    print(f"loaded {args.ckpt}, epoch = {ckpt.get('epoch')}")

    key_v = "net_v" if args.use_raw else "ema_v"
    key_s = "net_s" if args.use_raw else "ema_s"
    print(f"using {'raw' if args.use_raw else 'EMA'} weights")

    net_v = U_Net(128, 512).to(device)
    net_s = U_Net(128, 512).to(device)
    net_v.load_state_dict(ckpt[key_v])
    net_s.load_state_dict(ckpt[key_s])
    net_v.eval()
    net_s.eval()

    flow = FlowMatching(interp_func=linear_coeffs)

    torch.manual_seed(args.seed)
    x0 = torch.randn(args.n, 3, 32, 32, device=device)

    presets = build_presets()
    rows = []
    labels = []
    for label, kind, g_fn in presets:
        print(f"sampling: {label} ...")
        x = run_preset(flow, net_v, net_s, x0, args.steps, kind, g_fn)
        x = (x.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        rows.append(x)
        labels.append(label)

    padding = 2
    grid_input = torch.cat(rows, dim=0)
    grid = make_grid(grid_input, nrow=args.n, padding=padding)
    save_image(grid, args.out)

    add_row_labels(args.out, labels, img_h=32, padding=padding)

    print(f"\nsaved -> {os.path.abspath(args.out)}")
    print("rows (top to bottom):")
    for label in labels:
        print(f"  - {label}")


if __name__ == "__main__":
    main()
