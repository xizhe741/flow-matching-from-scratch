"""
Flow Matching 训练脚本 (CIFAR-10).

风格参考 diffusion_model_from_scratch/src/training/trainer.py:
- 使用 accelerate 做多卡 / 自动 device 处理
- AdamW + EMA
- 周期保存 checkpoint 到 checkpoints/latest.pt 和 checkpoints/epoch_xxxx.pt
"""

import copy
import os

import torch
import torchvision
import torchvision.transforms as T
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.flow.flow_matching import FlowMatching
from src.flow.interpolant import linear_coeffs, noise_magnify
from src.model.U_net import U_Net


# ---------- EMA ----------
class EMA:
    def __init__(self, model, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)


# ---------- Checkpoint ----------
def save_checkpoint(path, epoch, net_v, net_s, opt_v, opt_s, ema_v, ema_s, with_optimizer=True):
    """with_optimizer=False 时不存 optimizer state, ckpt 体积约减半 (适合周期性归档).
    latest.pt 用于断点续训, 必须存 optimizer;
    epoch_xxxx.pt 只用于事后采样, 不存 optimizer.
    """
    payload = {
        "epoch": epoch,
        "net_v": net_v.state_dict(),
        "net_s": net_s.state_dict(),
        "ema_v": ema_v.shadow.state_dict(),
        "ema_s": ema_s.shadow.state_dict(),
    }
    if with_optimizer:
        payload["opt_v"] = opt_v.state_dict()
        payload["opt_s"] = opt_s.state_dict()
    torch.save(payload, path)


def prune_old_ckpts(ckpt_dir, keep_last_n=3):
    """只保留最近 keep_last_n 个 epoch_xxxx.pt, 其余删掉. latest.pt 永远保留."""
    import glob
    files = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
    for f in files[:-keep_last_n] if len(files) > keep_last_n else []:
        try:
            os.remove(f)
            print(f"  pruned old ckpt: {f}")
        except OSError as e:
            print(f"  failed to prune {f}: {e}")


def load_checkpoint(path, net_v, net_s, opt_v, opt_s, ema_v, ema_s, device):
    ckpt = torch.load(path, map_location=device)
    net_v.load_state_dict(ckpt["net_v"])
    net_s.load_state_dict(ckpt["net_s"])
    ema_v.shadow.load_state_dict(ckpt["ema_v"])
    ema_s.shadow.load_state_dict(ckpt["ema_s"])
    # 不含 optimizer state 的 ckpt (epoch_xxxx.pt) 跳过 opt 恢复
    if "opt_v" in ckpt and opt_v is not None:
        opt_v.load_state_dict(ckpt["opt_v"])
        opt_s.load_state_dict(ckpt["opt_s"])
    return ckpt["epoch"]


# ---------- 训练 ----------
def main():
    accelerator = Accelerator()
    device = accelerator.device
    print(
        f"[rank={accelerator.process_index}/{accelerator.num_processes}] "
        f"local_rank={accelerator.local_process_index} "
        f"device={device} "
        f"LOCAL_RANK_env={os.environ.get('LOCAL_RANK')} "
        f"RANK_env={os.environ.get('RANK')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"torch.cuda.device_count()={torch.cuda.device_count()}",
        flush=True,
    )

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=384,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
    )

    net_v = U_Net(128, 512)
    net_s = U_Net(128, 512)
    opt_v = AdamW(net_v.parameters(), lr=2e-4)
    opt_s = AdamW(net_s.parameters(), lr=2e-4)
    ema_v = EMA(net_v, decay=0.9999)
    ema_s = EMA(net_s, decay=0.9999)
    ema_v.shadow.to(device)
    ema_s.shadow.to(device)

    flow = FlowMatching(interp_func=linear_coeffs, sigma_func=noise_magnify)

    net_v, net_s, opt_v, opt_s, dataloader = accelerator.prepare(
        net_v, net_s, opt_v, opt_s, dataloader
    )

    ckpt_dir = "checkpoints"
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 0
    ckpt_path = os.path.join(ckpt_dir, "latest.pt")
    # 如果想断点续训, 取消下面注释:
    # if os.path.exists(ckpt_path):
    #     start_epoch = load_checkpoint(
    #         ckpt_path,
    #         accelerator.unwrap_model(net_v),
    #         accelerator.unwrap_model(net_s),
    #         opt_v, opt_s, ema_v, ema_s, device,
    #     ) + 1
    #     if accelerator.is_main_process:
    #         print(f"Resumed from epoch {start_epoch}")

    train_epoch = 300
    for i in range(start_epoch, train_epoch):
        for images, _labels in dataloader:
            x_1 = images
            x_0 = torch.randn_like(x_1)

            loss_v, loss_s = flow.compute_loss(
                accelerator.unwrap_model(net_v),
                accelerator.unwrap_model(net_s),
                x_0, x_1,
            )

            opt_v.zero_grad()
            accelerator.backward(loss_v)
            opt_v.step()

            opt_s.zero_grad()
            accelerator.backward(loss_s)
            opt_s.step()

            ema_v.update(accelerator.unwrap_model(net_v))
            ema_s.update(accelerator.unwrap_model(net_s))

        if accelerator.is_main_process and (i % 10 == 9 or i == train_epoch - 1):
            print(f"epoch {i + 1} : loss_v = {loss_v:.4f}, loss_s = {loss_s:.4f}")
            # latest.pt: 完整 ckpt (含 optimizer), 用于断点续训
            save_checkpoint(
                ckpt_path, i,
                accelerator.unwrap_model(net_v),
                accelerator.unwrap_model(net_s),
                opt_v, opt_s, ema_v, ema_s,
                with_optimizer=True,
            )
            # epoch_xxxx.pt: 不含 optimizer, 只用于事后采样对比
            epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{i + 1:04d}.pt")
            save_checkpoint(
                epoch_ckpt, i,
                accelerator.unwrap_model(net_v),
                accelerator.unwrap_model(net_s),
                opt_v, opt_s, ema_v, ema_s,
                with_optimizer=False,
            )
            prune_old_ckpts(ckpt_dir, keep_last_n=3)


if __name__ == "__main__":
    main()
