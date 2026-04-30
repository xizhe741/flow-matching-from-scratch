"""
=============================================================================
 FID evaluation: minimal single-config scaffolding
=============================================================================
最小评估流程: 单一配置 (默认 SDE + ScaledSigma(c=1.0)), 计算 FID 并打印.

设计目标:
- 不落盘. torchmetrics.image.fid.FrechetInceptionDistance 直接接 uint8 tensor.
- 参数小: 默认 N=5000 张生成 + 完整 CIFAR-10 train 作为参考分布.
- 单种 g_fn: 默认 ScaledSigma(c=1.0); 后续要扩展时改 build_g_fn 即可.

依赖:
- torchmetrics[image]  (FrechetInceptionDistance, Inception V3 feature extractor)

Usage:
    python scripts/eval_fid.py --ckpt checkpoints/latest.pt
    python scripts/eval_fid.py --n 2000 --steps 100 --batch 64
=============================================================================
"""

import argparse
import os
import sys

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.flow.diffusion_coef import ScaledSigma
from src.flow.flow_matching import FlowMatching
from src.flow.interpolant import linear_coeffs
from src.model.U_net import U_Net


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float -> [0, 255] uint8, 形状不变."""
    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    return (x * 255.0).round().to(torch.uint8)


@torch.no_grad()
def collect_real(fid: FrechetInceptionDistance, batch_size: int, device, n_real: int):
    """喂 CIFAR-10 train 真实图. n_real=None 表示喂完整 train set (50000 张)."""
    transform = T.ToTensor()  # [0, 1] float
    dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    seen = 0
    for images, _ in loader:
        images = images.to(device)
        images_u8 = (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        fid.update(images_u8, real=True)
        seen += images.shape[0]
        if n_real is not None and seen >= n_real:
            break
    print(f"[real] fed {seen} CIFAR-10 train images to FID")


@torch.no_grad()
def collect_fake(
    fid: FrechetInceptionDistance,
    flow: FlowMatching,
    net_v,
    net_s,
    n_fake: int,
    batch_size: int,
    steps: int,
    device,
    g_fn,
):
    """采样 n_fake 张生成图喂 FID. 每个 batch 用 sde_sample 跑."""
    generated = 0
    while generated < n_fake:
        b = min(batch_size, n_fake - generated)
        x0 = torch.randn(b, 3, 32, 32, device=device)
        x, _ = flow.sde_sample(net_v, net_s, x0=x0, steps=steps, g_fn=g_fn)
        fid.update(to_uint8(x), real=False)
        generated += b
        if generated % (batch_size * 10) == 0 or generated >= n_fake:
            print(f"[fake] sampled {generated}/{n_fake}")
    print(f"[fake] fed {generated} generated images to FID")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/latest.pt")
    parser.add_argument("--n", type=int, default=5000, help="生成图张数")
    parser.add_argument("--n-real", type=int, default=None,
                        help="参考真实图张数 (默认 None = 完整 50000 train)")
    parser.add_argument("--batch", type=int, default=128, help="采样 batch")
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
    g_fn = ScaledSigma(c=1.0)
    print(f"sampler: SDE, g_fn = ScaledSigma(c=1.0), steps = {args.steps}")

    torch.manual_seed(args.seed)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    collect_real(fid, batch_size=args.batch, device=device, n_real=args.n_real)
    collect_fake(
        fid, flow, net_v, net_s,
        n_fake=args.n, batch_size=args.batch, steps=args.steps,
        device=device, g_fn=g_fn,
    )

    score = fid.compute().item()
    print(f"\nFID = {score:.4f}")
    print(f"  config: SDE + ScaledSigma(c=1.0), steps={args.steps}, "
          f"n_fake={args.n}, n_real={args.n_real or 50000}")


if __name__ == "__main__":
    main()
