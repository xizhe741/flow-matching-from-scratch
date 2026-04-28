"""
Flow Matching 训练脚本 (CIFAR-10).

风格参考 diffusion_model_from_scratch/src/training/trainer.py:
- 使用 accelerate 做多卡 / 自动 device 处理
- AdamW + EMA
- 周期保存 checkpoint 到 checkpoints/latest.pt 和 checkpoints/epoch_xxxx.pt
- wandb 监控: loss_v / loss_s / grad_v / grad_s 时间序列, 自动追踪 GPU 系统指标
"""

import copy
import os
from datetime import datetime

import torch
import torchvision
import torchvision.transforms as T
import wandb
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.flow.flow_matching import FlowMatching
from src.flow.interpolant import linear_coeffs
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

    # 训练超参集中放, 顺便给 wandb config 用
    cfg = {
        "model": "U_Net(128, 512)",
        "interpolant": "linear",
        "score_target": "-z (sigma * s parameterization)",
        "batch_size": 320,
        "lr_v": 1e-4,
        "lr_s": 1e-4,
        "ema_decay": 0.99,
        "max_grad_norm": 1.0,
        "train_epoch": 200,
        "num_processes": accelerator.num_processes,
    }

    # wandb 初始化, 仅 main process 上报
    if accelerator.is_main_process:
        wandb.init(
            project="flow-matching",
            name=f"run-{datetime.now().strftime('%Y%m%d-%H%M')}",
            config=cfg,
            resume="allow",
        )

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
    )

    net_v = U_Net(128, 512)
    net_s = U_Net(128, 512)
    opt_v = AdamW(net_v.parameters(), lr=cfg["lr_v"])
    opt_s = AdamW(net_s.parameters(), lr=cfg["lr_s"])
    ema_v = EMA(net_v, decay=cfg["ema_decay"])
    ema_s = EMA(net_s, decay=cfg["ema_decay"])
    ema_v.shadow.to(device)
    ema_s.shadow.to(device)

    flow = FlowMatching(interp_func=linear_coeffs)

    net_v, net_s, opt_v, opt_s, dataloader = accelerator.prepare(
        net_v, net_s, opt_v, opt_s, dataloader
    )

    ckpt_dir = "checkpoints"
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 0
    ckpt_path = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(ckpt_path):
        start_epoch = load_checkpoint(
            ckpt_path,
            accelerator.unwrap_model(net_v),
            accelerator.unwrap_model(net_s),
            opt_v, opt_s, ema_v, ema_s, device,
        ) + 1
        if accelerator.is_main_process:
            print(f"Resumed from epoch {start_epoch}")

    train_epoch = cfg["train_epoch"]
    steps_per_epoch = len(dataloader)
    log_every = 50  # 每 50 step 写一次 wandb

    for i in range(start_epoch, train_epoch):
        for step_idx, (images, _labels) in enumerate(dataloader):
            x_1 = images
            x_0 = torch.randn_like(x_1)

            loss_v, loss_s = flow.compute_loss(net_v, net_s, x_0, x_1)

            loss = loss_v + loss_s
            opt_v.zero_grad()
            opt_s.zero_grad()
            accelerator.backward(loss)
            grad_v = accelerator.clip_grad_norm_(net_v.parameters(), max_norm=cfg["max_grad_norm"])
            grad_s = accelerator.clip_grad_norm_(net_s.parameters(), max_norm=cfg["max_grad_norm"])
            opt_v.step()
            opt_s.step()

            ema_v.update(accelerator.unwrap_model(net_v))
            ema_s.update(accelerator.unwrap_model(net_s))

            global_step = i * steps_per_epoch + step_idx
            if accelerator.is_main_process and global_step % log_every == 0:
                wandb.log(
                    {
                        "loss/v": loss_v.item(),
                        "loss/s": loss_s.item(),
                        "grad/v": grad_v.item(),
                        "grad/s": grad_s.item(),
                        "epoch": i + step_idx / steps_per_epoch,
                    },
                    step=global_step,
                )

        if accelerator.is_main_process and (i % 10 == 9 or i == train_epoch - 1):
            print(
                f"epoch {i + 1} : "
                f"loss_v = {loss_v:.4f}, loss_s = {loss_s:.4f}, "
                f"grad_v = {grad_v:.3f}, grad_s = {grad_s:.3f}"
            )
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

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
