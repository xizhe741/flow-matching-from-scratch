# 待答问题清单 (TECHNICAL.md 第 2 节剩余 + 关键概念)

按照"我提问 → 你答 → 我判断 → 写进 TECHNICAL.md"的流程进行。
不会的就直说"不知道"或"模糊",我打回让你再想或者补充背景。

---

## Q5: src/flow/flow_matching.py — 损失函数

### Q5a: 速度损失 L_v 的物理含义

```python
loss_v = ((v_pred - xdot_t) ** 2).mean()
```

1. `xdot_t = α̇·x_1 + β̇·x_0 + σ̇·z` 物理上是什么? (提示: dx_t/dt)
2. 为什么让网络拟合这个量是合理的? 反向采样时我们用 v 干什么?
3. 训练时知道真值,采样时不知道——v 是怎么"泛化"的?

### Q5b: 分数损失 L_s 为什么这样写

```python
loss_s = ((sigma * s_pred + z) ** 2).mean()
```

1. 真实的 score `∇log p_t(x_t)` 算得出来吗? 算不出来的话,我们到底在拟合什么?
2. 为什么 target 是 `-z/σ` 而不是别的形式?
   (提示: 条件分布 `x_t | x_0, x_1 ~ N(α·x_1 + β·x_0, σ²·I)` 的 log-density 对 x_t 求导)
3. 端点 σ→0 时这个 loss 退化成什么? 为什么要用 Beta(2,2) 避开端点?
   (这个之前答过, 回顾用)

---

## Q6: 反向 SDE 采样

```python
drift = (v + 0.5 * sigma_t**2 * s) * dt
diffusion = sigma_t * (dt**0.5) * torch.randn_like(x)
```

1. 为什么 drift 里要有 `0.5·σ²·s` 这一项? 只用 v 不行吗? 这一项叫什么?
2. 为什么 diffusion 是 `σ·√dt·z` 而不是 `σ·dt·z`?
   (提示: 布朗运动的标度律 `dW ~ √dt · N(0, I)`)
3. 最后一步为什么要让 diffusion = 0?

---

## Q7: src/model/U_net.py + modules.py — 网络细节

1. `sinusoidal_embedding` 把标量 t 映射到 256 维,为什么不直接用一个 Linear?
2. ResBlock 里为什么用 GroupNorm 而不是 BatchNorm?
3. self_attention 加在中间两层 (不是所有层),为什么?
4. forward 里 `if t.dim() > 1: t = t.view(-1)` 这行兼容什么场景?

---

## Q8: src/training/trainer.py — 训练循环

1. EMA 的 `decay = 0.9999` 意味着什么? 有效"记忆窗口"是多少 step?
   (这个之前讨论过, 回顾用)
2. AdamW 比 Adam 多了什么? 为什么 diffusion / flow matching 类项目通常用 AdamW?
3. `accelerator.prepare(net_v, net_s, opt_v, opt_s, dataloader)` 一次接收 5 个对象,
   分别对它们做了什么?
4. 为什么 `train_epoch = 300` 而不是更多 / 更少? 怎么判断训练是否充分?

---

## Q9: ⚠️ 重点 —— DDP / unwrap_model 机制

这是你训练 bug 的根源,必须搞清楚。

1. `accelerator.prepare(net_v)` 把网络包成什么? DDP 的全称和工作原理是什么?
2. **forward 时的 hook**: DDP 在 forward 调用时挂了什么? backward 时这些 hook 干什么?
3. `accelerator.unwrap_model(net_v)` 返回的对象和原始 `net_v` 有何区别?
4. 为什么以下三种调用必须区分:
   - `compute_loss(net_v, ...)` ← **不能 unwrap** (forward 必须触发 DDP hook)
   - `ema.update(unwrap_model(net_v))` ← **必须 unwrap** (EMA 不需要梯度同步)
   - `save_checkpoint(unwrap_model(net_v).state_dict())` ← **必须 unwrap** (避免 `module.` 前缀)
5. 如果在 `compute_loss` 里 unwrap, 实际会发生什么? 两张卡梯度怎么处理? 损失为什么会震荡不下降?
6. `accelerator.backward(loss)` 比 `loss.backward()` 多做了什么?
   为什么开 DDP / 混合精度后必须用前者?

---

## Q10: scripts/sample_ckpt.py + compare_sampling.py — 采样脚本

1. 为什么采样时要 `clamp(-1, 1)` 再 `(x + 1) / 2`?
   训练数据是怎么归一化的?
2. `make_grid(x, nrow=8)` 是把 64 张图拼成几行几列?
3. `compare_sampling.py` 里为什么要固定 `torch.manual_seed(args.seed)`?
   (提示: 4 种方案要在同一组初始噪声 x_0 上跑)
4. ODE 采样和 SDE 采样的命令行开关如何映射到代码里的不同分支?

---

## 第 3 节: 技术分析 (后续讨论)

预留给训练完成后写。需要包含:
- ODE vs SDE 取舍 (附 compare.png 的实测对比)
- 训练精度选择 (fp32 / fp16 / bf16, 你 diffusion 时遇到 fp16 没学到的坑)
- 采样步数取舍 (200 vs 500 vs 1000)
- 显存 / batch / 多卡踩坑 (本次 OOM 历史)
- 为什么 σ(t) = √(2t(1-t)) 这个特定形式 (Albergo 论文的选择)
- v 网络 + s 网络联合训练的 lr / batch 调参经验

---

## 第 4 节: AI 协作复盘 (最后写)

需要你亲自给原始素材的部分。表格在之前对话里给过:
对每个模块/概念评 "理解度 1-5" + "出 bug 能不能独立 debug"。
回 app 里我会把这张表再贴一遍, 你逐行答。
