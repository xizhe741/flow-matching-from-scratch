# 项目状态 (2026-04-28)

下次会话的入口点。读完这份就有完整上下文。

## 项目目标

CIFAR-10 上从零实现 stochastic flow matching, 联合训练 velocity 网络 v 和 score 网络 s, 反向 SDE 采样。参考 Albergo et al. 2023, 项目结构对照 `diffusion_model_from_scratch/`。

## 当前状态

**训练正在进行**, run 名 `run-20260428-1526` (wandb 上能看到)。

### 已完成

#### 项目搭建 (用户主导, AI 协助)
- 完整目录结构: `src/{flow,model,training}/` + `scripts/`
- pyproject.toml + .gitignore + README 占位
- GitHub 仓库已创建并 push 到 https://github.com/xizhe741/flow-matching-from-scratch
- 数据集 CIFAR-10 已下载到 `data/`

#### 核心代码 (用户主导写过 MNIST 单文件版, AI 帮搬到项目结构)
- `src/flow/interpolant.py` — 插值方案 + σ(t) + interp 函数
- `src/flow/flow_matching.py` — `FlowMatching` 类: compute_loss + sde_sample + ode_sample
- `src/model/U_net.py` + `modules.py` — U-Net (照搬 diffusion 项目, 3ch 32×32)
- `src/training/trainer.py` — 训练循环 + EMA + checkpoint + accelerate 多卡 + wandb
- `scripts/sample_ckpt.py` — 加载 ckpt 反向 SDE/ODE 采样
- `scripts/compare_sampling.py` — 4 种采样方式 (EMA/raw × SDE/ODE) 对比

#### 关键 bug 修复 (AI 主导诊断, 用户审核确认)
1. **DDP forward 用了 unwrap_model** → 修复: forward 用 DDP 包装的 net
2. **双反传两份计算图占显存** → 修复: 合并为 `(loss_v + loss_s).backward()`
3. **score loss target 是 -z/σ 端点爆炸** → 修复: 改成预测 -z (DDPM ε-prediction 风格)
4. **缺少梯度裁剪** → 修复: `accelerator.clip_grad_norm_(..., max_norm=1.0)`
5. **xdot_t 含 σ̇·z 干扰项, 训练 target 方差大** → 修复: 拿掉 σ̇·z
6. **sinusoidal_embedding 在 t∈[0,1] 高频项失效** → 修复: t * 1000.0 后再嵌入

#### 训练监控
- wandb 集成完成, 实时上传 loss/v, loss/s, grad/v, grad/s
- wandb URL: https://wandb.ai/fengxz24-tsinghua-university/flow-matching

#### 文档
- `TECHNICAL.md` 已写第 1, 2, 3, 4 节
  - 第 1 节: 项目框架 (目录树 + 数据流图 + 配置)
  - 第 2 节: 每份代码做什么 / 怎么实现 (interpolant.py, flow_matching.py 已写)
  - 第 3 节: 技术分析 (复用代码 ≠ 正确, target 方差分析, ODE/SDE 取舍, 监控价值)
  - 第 4 节: AI 协作复盘 (6 条经验)
- `QUESTIONS.md` 已写, 含 Q5-Q10 问题清单
- `PROJECT_STATUS.md` (本文件)

#### 已答的问答 (Q5, Q6 部分)
- Q5a: 速度损失 L_v — 条件速度场 vs 边际速度场, 连续性方程 ✅
- Q5b: 分数损失 L_s — 真分数算不出来, 用条件分数 + 改预测 -z ✅
- Q6: 反向 SDE — Itô 修正项, √dt 标度律, 最后一步置零 ✅

### 进行中

- **训练 run-20260428-1526**: 修完 6 个 bug 后从零重训
  - 当前 step ~750 (约 epoch 5)
  - loss/v ≈ 0.74 (从 1.4 起步), loss/s ≈ 0.38 (从 1.1 起步)
  - 单步波动 ±0.05 (修前是 ±0.3, **改善 6 倍**)
  - 状态: 健康, 进入第一个平台期
  - 预期: epoch 20 内 loss/v 应继续降到 < 0.7

### 未完成

#### TECHNICAL.md 第 2 节剩余 (按 QUESTIONS.md)
- Q7: U-Net 细节 (sinusoidal embedding / GroupNorm / attention 位置 / t.dim() 兼容)
- Q8: trainer 循环 (EMA decay / AdamW vs Adam / accelerator.prepare 做什么 / 训练终止判据)
- Q9: ⚠️ DDP / unwrap_model 机制 (用户重点关注, 这是这次主要 bug 来源)
- Q10: 采样脚本 (clamp+归一化 / make_grid / seed)

#### TECHNICAL.md 第 3 节扩充
- 训练精度选择 (fp32/fp16/bf16, 用户 diffusion 时遇过 fp16 underflow)
- 采样步数取舍 (200 vs 500 vs 1000)
- 显存 / batch / 多卡踩坑历史
- 为什么 σ(t) = √(2t(1-t)) 的特定形式

#### 训练完成后的工作
- 等训练跑到 epoch 100-200 (预计还要几小时)
- 用最终 ckpt 跑 `compare_sampling.py` 4 种采样对比
- 把生成图加到 `assets/`
- TECHNICAL.md 第 1 节加上"实现效果"小节(参考 diffusion 项目的写法)

#### 同步代码到 GitHub
- 现在云端已有大量改动, 本地 trainer.py 也改了, 需要 commit 一波
- 同步前先确认: 本地 / 云端代码是否完全一致

#### 可选改进 (按优先级)
- [ ] EMA decay 调优 (当前 0.99, 可能太低 / 太高)
- [ ] lr scheduler (warmup + cosine decay)
- [ ] 把 σ̇·z 拿掉的改动写进 interpolant.py 的 docstring 解释
- [ ] 删 `assets/` 下旧的占位图(如果有)
- [ ] 整理出最终 README.md

## 责任分工概览

| 模块 | 用户 | AI |
|---|---|---|
| MNIST 单文件版原型 | ✅ 主写 | — |
| 项目目录结构 | 决策 | 搭建 |
| interpolant.py / flow_matching.py 代码 | 审核 | 主写 (基于用户原型) |
| U-Net 模型 | — | 复用 diffusion 项目 |
| trainer.py | 审核 + 改 lr/batch | 主写 (基于 diffusion 项目改造) |
| Bug 1-6 诊断 | 实测反馈 | 主导分析 + 修复方案 |
| 训练监控 (wandb) | 配置账号 | 集成代码 |
| 训练运行 + 调试 | ✅ 实际操作 | 在线指导 |
| TECHNICAL.md | 审核 + 用自己语言答 Q5/Q6 | 起草框架 + 整合用户回答 |

## 立即可做

下一次会话开始时, 按这个顺序:

1. **看训练状态** — 打开 wandb URL 看 run-20260428-1526 是否还在跑, 当前 epoch / loss
2. **如果训练崩了** — 看 wandb Logs 标签, 把 traceback 给 AI 排查
3. **如果训练在跑** — 继续 Q7-Q10 的问答, 把 TECHNICAL.md 第 2 节剩余写完
4. **训练完成后** — 跑 compare_sampling 看图, 写"实现效果"小节, 提交 GitHub

## 重要文件位置

- 本地项目根: `/mnt/e/VScode/py-codes/mlstudying/flow_matching_from_scratch/`
- 云端项目根: `/root/shared-nvme/flow_matching_from_scratch/`
- wandb run: https://wandb.ai/fengxz24-tsinghua-university/flow-matching
- GitHub: https://github.com/xizhe741/flow-matching-from-scratch

## 关键历史决策

- 数据集选 CIFAR-10 而非 MNIST: 复用 diffusion 项目的 U-Net (3ch 32×32) 不用改
- batch=320, 双卡 DDP: 经历过 OOM 后从 384 降下来
- lr=1.5e-4 (双网络相同): 待 epoch 50 后看是否调整
- ema_decay=0.99 (不是常用的 0.9999): 用户改的, 想让 EMA 追得快点

## 已知风险

- **EMA decay=0.99 可能让 EMA 抖动大**: 后续如果 EMA 采样质量不稳, 改回 0.999 试试
- **GitHub 同步缺失**: 当前 GitHub 上还是初始 commit, 后续修复都没 push
- **磁盘紧张**: 50 GB 总容量, ckpt 累积要看好 prune 是否生效
