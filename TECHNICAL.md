# Flow Matching from Scratch — 技术文档

CIFAR-10 上从零实现 stochastic flow matching 的复盘记录。本文档不展开 flow matching 的数学背景，专注于工程实现里实际遇到的问题与决策。

---

## 1. 项目框架

### 目录结构

```
flow_matching_from_scratch/
├── pyproject.toml                # 包元信息, 用 pip install -e . 让 src 可被导入
├── README.md                     # (留空, 待补)
├── .gitignore
├── assets/                       # 采样图 / 可视化输出
├── checkpoints/                  # 训练保存的 .pt 文件
├── src/
│   ├── flow/                     # flow matching 算法本体 (与数据/网络解耦)
│   │   ├── interpolant.py        # 插值系数 α(t)/β(t) 与噪声放大 σ(t)
│   │   └── flow_matching.py      # loss 计算 + 反向 SDE 采样器
│   ├── model/                    # 速度场 / 分数场共用的 U-Net
│   │   ├── modules.py            # ResBlock / self-attention / 时间嵌入 / 上下采样
│   │   └── U_net.py              # U-Net 主体 (照搬 diffusion 项目)
│   └── training/
│       └── trainer.py            # 训练循环 + EMA + checkpoint, 走 accelerate 多卡
└── scripts/
    └── sample_ckpt.py            # 加载 ckpt, 从噪声反向 SDE 采样, 存成 8x8 网格
```

### 数据流

**训练**:
```
x_1 (CIFAR-10 真实图)  ─┐
                       ├─→ interp(x_0, x_1, t) ─→ (x_t, xdot_t, σ, z)
x_0 ~ N(0, I)         ─┘                           │
                                                   ├─→ net_v(x_t, t) → v_pred → L_v = ‖v_pred − xdot_t‖²
                                                   └─→ net_s(x_t, t) → s_pred → L_s = ‖σ·s_pred + z‖²
```

**采样** (反向 SDE):
```
x_0 ~ N(0, I)
for t in 0, dt, 2dt, ..., 1:
    drift     = (v(x_t, t) + 0.5·σ(t)²·s(x_t, t)) · dt
    diffusion = σ(t) · √dt · ε,  ε ~ N(0, I)        # 最后一步置零
    x_{t+dt} = x_t + drift + diffusion
得到 x_1
```

### 关键配置

| 部件 | 配置 |
|---|---|
| U-Net | base_channels=128, embedded_dim=512, 4 级 encoder/decoder, 中间两级带 self-attention |
| 插值方案 | linear: α(t)=t, β(t)=1−t |
| 噪声放大 | σ(t) = √(2t(1−t)) |
| 时间采样 | Beta(2, 2), 截断到 [0.01, 0.99] |
| 优化器 | AdamW, lr=2e-4, 双网络各一份 |
| EMA | decay=0.9999, 双网络各一份 shadow |
| 训练 | 300 epoch, batch=384, accelerate 多卡 |
| 采样 | 反向 SDE, steps=200 |
| 数据 | CIFAR-10, T.Normalize((0.5,)*3, (0.5,)*3) → [-1, 1] |

### 入口命令

```bash
pip install -e .                                    # 一次性, 让 src 可被导入
accelerate launch --num_processes=2 -m src.training.trainer    # 训练
python scripts/sample_ckpt.py --n 64 --steps 200    # 采样
```

---

## 2. 每份代码做什么 / 怎么实现

### 2.1 [src/flow/interpolant.py](src/flow/interpolant.py)

负责定义"从 x_0 到 x_1 的带噪声插值轨迹"以及它对 t 的导数。整个文件不依赖任何网络结构，是纯粹的数学工具。

#### `noise_magnify(t)` — 噪声放大函数 σ(t)

设计动机:flow matching 的反向过程不仅要有**漂移项**(由速度场 v 提供方向),也要有**扩散项**(注入随机性,避免模式坍缩)。这要求前向插值时在每一步都叠加一个正态噪声 z,得到带噪声的插值轨迹:

```
x_t = α(t)·x_1 + β(t)·x_0 + σ(t)·z,    z ~ N(0, I)
```

噪声幅度 σ(t) 不能在端点处也很大 —— 在 t=0 时 x_t 应该等于 x_0(纯噪声本身),在 t=1 时 x_t 应该等于 x_1(干净图像),两端再叠噪声会破坏边界条件。所以 σ(t) 需要是一条**两端为 0、中间最大**的钟形曲线。本项目沿用 Albergo et al. 的选择:

```
σ(t) = √(2t(1-t))     # t=0/t=1 时为 0, t=0.5 时取最大值 1
```

函数同时返回 σ 对 t 的导数 σ̇,因为整条插值轨迹对 t 求导后会用到:

```
xdot_t = α̇(t)·x_1 + β̇(t)·x_0 + σ̇(t)·z
```

这里 `(α̇, β̇, σ̇)` 各自是对应系数对 t 的导数,分别乘以 `(x_1, x_0, z)`,这就是训练时速度场要拟合的"真值"。

**端点数值问题**:σ̇(t) = (1−2t)/√(2t(1−t)) 在 t=0、t=1 时分母为 0,会得到 NaN。`endpoint_mask` 在这两点把 σ 和 σ̇ 强制设为 0,绕开除零。训练时 t 从 [0.01, 0.99] 采样本身就避免了端点,但 mask 是兜底。

#### `linear_coeffs(t)` 和 `trig_coeffs(t)` — 两种插值方案

把上面的公式参数化:

| 方案 | α(t) | β(t) | α̇(t) | β̇(t) |
|---|---|---|---|---|
| `linear_coeffs` | t | 1−t | 1 | −1 |
| `trig_coeffs` | sin(πt/2) | cos(πt/2) | (π/2)cos(πt/2) | −(π/2)sin(πt/2) |

线性方案最简单,边界条件 α(0)=0、α(1)=1 自动满足;三角方案在端点的导数为 0,过渡更"光滑"。本项目默认用 linear。

#### `interp(x_0, x_1, t, ...)` — 一次给出 x_t 和 xdot_t

把上面所有零件拼起来:采一个 z,算出 x_t 和 xdot_t,顺便返回当前 σ 和 z(后面 score loss 要用)。这是训练循环里调用的统一入口。


## 3. 技术分析

> _待补_

## 4. AI 协作复盘

> _待补_
