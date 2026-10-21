# tour-mtp：多地平线监督（multi-horizon supervision）

## 机制

来源：用户参考模型 `3120012elite.py` 的 MTP（multi-token prediction）。

本实验采用**共享 head + 可学地平线偏移**的轻量形式（隔离"训练信号"变量，不引入额外读出容量）：

```
logits2 = (sem + d_mtp) · headᵀ + bout      # 预测 t+2
e2      = softmax(logits2) − onehot(x_{t+2}) # 同一套掩码归一化
Ls     += λ · (e2·head ⊙ σ'(sem)) · Wp       # 与主信号共用状态迹
g_head += λ · e2ᵀ·(sem + d_mtp)
∇d_mtp  = λ · Σ_b (e2·head)
```

- 新增参数：仅 `d_mtp`（M 维，零初始化）。
- 状态迹（eK/e_bg/e_bd）**不需要新增账本**：两个地平线共用同一条精确迹，
  只是学习信号从 `e` 变为 `e + λ·g_sem2`。
- 关闭时与基线逐位一致（smoke：500 流 / K2 / 100 步 → 5.690153，与历史一致）。
- 另备 **MTP_SEP=1**（独立 head2/bout2，DeepSeek 式）：用于区分"多地平线信号"与"新增读出容量"；
  独立头增加 V·M 参数，主实验先用共享头隔离信号。

## 为什么关注 MTP（用户校正）

MTP 的首要目的不是加速，而是**消除常数解**：若只监督 t+1，隐状态塌缩为常量也能压 loss（死通道）；
增设 t+2 目标后，常量状态无法同时预测两个不同步的 token，从而被迫保留随时间变化的动力学。
因此评估除 ppl 外，还必须测量**状态时间方差与常数态读出 CE**（见 diag_collapse.py）。
- 已知近似：t+2 的监督信号作用在当前时刻的迹上（一步延迟），属在线局部规则的常规折中。

## 协议

- GPU（RTX 5080），H=256 / D=192 / M=192，K=1，TAU=1，η=2e-3
- 1024 流 × 20k 步，3 种子（43/44/45），检查点 10k / 20k
- 对照：MTP=0（同配置基线）vs MTP=1，λ=0.5

## 结果

待 Round 29 完成（见上方 runner 输出 / TOURNAMENT.md）。

## 复现

```bash
python experiments/tour_mtp_run.py
```
