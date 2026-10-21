# BioELM — 生物局部规则语言模型（零 autograd）

**A backprop-free language model trained purely with biologically plausible local learning rules.**

[![Release](https://img.shields.io/github/v/release/X005-0001-0001/BioELM?label=release)](https://github.com/X005-0001-0001/BioELM/releases)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

BioELM 是一个**训练全程零 autograd（代码中绝无 `loss.backward()`）**的语言模型。
学习完全由局部可获得的三因子信号驱动：选择性门控对角 SSM 上的**精确在线 RTRL 资格迹** +
元 Adam 在线更新。它要回答一个直接的问题：

> **不用全局反传，语言模型能走多远、能多快？**

在同一个对话 QA 语料、同一块硬件、同一墙钟预算下（只用账面参数，不玩虚的），
BioELM 对标准 TinyGPT2（正常反传训练）的对比见下表。

---

## English Abstract

BioELM is a language model **trained without any autograd / backpropagation**. Its circuit is a
selectively-gated diagonal SSM whose state admits *exact* online RTRL eligibility traces, updated by
three-factor local rules and a meta-Adam optimizer. Under identical wall-clock / data / hardware
budgets it is compared against a standard TinyGPT2 trained with backpropagation. The whole model is
released as a **single self-contained Python file** with a Chinese interactive menu (train / chat /
RL / GRPO / rejection sampling / long-dialogue splitting). Versions: see
[Releases](https://github.com/X005-0001-0001/BioELM/releases) (each release receives a Zenodo DOI).

---

## 实测结果（可复现，见 `experiments/`）

同一语料（`dsr1_v5_0001.jsonl`，3000 条对话，记录级 90/10 切分，仅 assistant 区间受训、
去重与泄露检查通过，词表 2754）：

| 对决 | BioELM | 对手（TinyGPT2，标准反传） | 条件 |
|---|---|---|---|
| 对话 QA 验证 ppl | **36.69** | 61.05 | CPU 300s 同墙钟 |
| 对话 QA 验证 ppl | **~16** | — | RTX 5080 240s，512 流 |
| 长程依赖竞技场（crit-CE） | ID **0.0069** / OOD **0.7555** | ID 0.9933 / OOD 2.1333 | 同墙钟同设备 |

GPU 对照（RTX 5080，240s 同墙钟，512 训练流，val 全量掩码 ppl）：

| 分支 | 训练模式 | 240s 步数 | 验证 ppl |
|---|---|---|---|
| elite 1.1.5 | 单层（h=512/d=128/m=128） | 55,600 | **15.54**（历史流水线口径） |
| elite 1.1.6 | 单层（h=256/d=192/m=192，新比例） | 56,972 | **15.77** |
| deep 1.1.0 | 循环 K=2 | 52,608 | 15.78 |
| elastic 1.0.0 | 一致性 KMAX=4 | 19,632 | 25.38 |

> 1.1.5 与 1.1.6 的直接受控 A/B（同脚本、同种子、全量 val）：
> 240s 16.07 → **15.77**；固定 20k 步 17.31 → **17.01**；受控 eager 吞吐 246–253 → 252–261 步/秒（持平）。
> 旧 15.54 来自另一条 20 轮端到端流水线，口径不同不可与 A/B 混比。
> 本轮 CPU 筛选与 GPU 高分辨率复验见 `experiments/TOURNAMENT.md`（Round 11–28）。

elastic 诚实说明：consist 训练 ≈2.8× 单层成本（每步完整+捷径双路线）；600s 延长训练后
K=2 ppl 22.79、K=4 ppl 21.57。其价值在**推理预算弹性**（一条模型、1..6 深度任意切换），
不是同墙钟训练质量——同预算要追平需更大步数或更便宜的训练模式。

吞吐（RTX 5080，512 训练流）：1.1.5 h=512/d=128/m=128 ≈ 97 万参数；1.1.6 h=256/d=192/m=192 ≈ 126 万参数，受控 eager A/B 两者持平（约 250 步/秒）。历史提升：
1.1.0 = 214 → 1.1.1 = 282 → 1.1.2 = 323 → 1.1.3 = 569 步/秒（291k tok/s；评估 92k tok/s）。
**1.1.5 实测 ≈ 699 步/秒**（20 轮端到端含每轮全量评估，CUDA Graph 整步回放；[O11-修] 移除逐步范数归约）。
**本版 1.1.5 = 自校准内核**（O11–O14 + O11-修）：自适应开销在 GPU 上可忽略，吞吐再上台阶；
核心变化是**免手调**——学习率由模型自身尺度（参数 RMS × 无量纲 ρ）派生并被损失反馈保守调速，
换模型规模/换奖励形状不改配置（CPU 三尺度验证：同配置落在各尺度手调最优 ±0.04 CE 内）。
1.1.5 修复 [O11] 的基准自反馈：步长基准 R 改为**建立时冻结**（载入 brain.pt 时刷新），
杜绝"权重漂移 → R 变大 → 步长变大"的正反馈（实测 3000 学习步 R 0.41→1.12）。

深度分支（deep 1.1.x）：[D1] **循环深度**——同一 token 积分 K 次后再读出/学习，权重共享、零新增参数，
由精确在线 RTRL 原生覆盖。CPU 同协议 500 条：K=1: 4.599 → K=2: 4.453 → K=4: 4.299（val CE）；
同期实测**层监督堆叠**（L2: 4.713）与**反馈对齐**（L2: 5.659）均劣于单层——深度靠"时间上的循环"
而不是"参数上的堆叠"（见下方 deep 版本表与 `experiments/deep_bench.py`）。
1.1.0 新增 [D2] **子步时间条件注入**（K≥3 自动启用）：每子步门控偏置 gb[k]/db[k]，精确局部规则学习。
GPU h=512 20k 步同机对照：K=3 2.8252→2.8020、K=4 2.8520→2.8012（CE），并逆转深度退化
（1.0.x 的 K2→K4 变差 2.8045→2.8520，1.1.0 K3/K4 稳定在 2.80）；K=2 中性且付 ~6% 吞吐 → 自动关闭并逐位退回 1.0.x。

弹性深度分支（elastic 1.x）：[E1] **弹性深度**——一条模型，推理预算 K 任意指定（1..K_MAX）、无需重训。
1.1.0 起默认训练模式为**深度随机化**（random：每步随机深度仅 CE，单路线）。
GPU 固定步数口径（RTX 5080，h=512，20k 步，val 掩码 CE）：

| 模式 | @K=1 | @K=2 | @K=3 | @K=4 | 每步成本 |
|---|---|---|---|---|---|
| **random（默认）** | **3.002** | **2.846** | **2.821** | 2.847 | ~1.07× |
| fixed K4 | 3.980 | 3.135 | 2.879 | **2.822** | ~1.5× |
| consist λ=1.0 | 3.306 | 3.184 | 3.120 | 3.113 | ~2.5× |

240s 同墙钟（random）：52.6k 步 → @K=1 ppl 18.4 / @K=2 **15.86** / @K=3 **15.68**（同墙钟 elite 单层 15.54）。
注：LoopFormer 式**一致性蒸馏**（consist）实为**短训练加速器**：3k 步内全档更优
（h=192：K1/K2 3.560/3.485 vs random 3.685/3.508），20k 步渐近反超
（K2 3.315 vs 3.140）；h=192 与 h=512 同向——决定变量是**训练时长**而非模型宽度
（冷热启动对照见 `expr-horizon` 分支），故保留为短预算/微调场景选项。
复现：`experiments/bench_elastic.py`（小尺度）与 `experiments/bench_elastic_gpu.py`（大尺度）。

复现口径与实验矩阵见 [`experiments/README.md`](experiments/README.md)。

---

## 架构一句话

```
g = σ(x·Wg + bg)          # 选择性写入门
d = σ(x·Wd + bd)          # 选择性地漏（[O12] 初始半衰期 √H 步：d0=1−ln2/√H，近似恒定记忆）
s_t = d ⊙ s_{t-1} + g ⊙ (B·x)     # 对角线性状态推进
sem = σ(s·Wp + bp)        # 读出瓶颈
logits = sem·headᵀ + bout
```

* 对角结构让**精确在线 RTRL 资格迹**封闭可算：三条 (H, d) 迹张量逐列递推，内存 O(B·H·d)。
* 学习信号 `e`（预测误差/奖励权重）经迹与本地雅可比直接得到各参数局部梯度——**无反向图**。
* 元 Adam（RMS 型）逐参数归一化在线更新 + 权重衰减；ATP 能量记账全程 GPU 侧（无逐步同步）。
* 铁律：训练侧零 autograd；对手 TinyGPT2 使用标准反传；对决必须同墙钟 / 同数据 / 同设备。

---

## 快速开始

```bash
pip install -r requirements.txt

# Windows / 通用
python bioelm_elite_1.1.6.py

# Linux / WSL（带 LINUX_* 加速块：分配器 / 线程 / TF32 / 实验性单步编译，均可关）
python bioelm_elitelinux_1.1.6.py

# 深度分支（循环深度探索线 1.1.x；内核 = 自校准 1.1.5 + [D1] 循环深度 + [D2] 子步时间条件）
python bioelm_deep_1.1.0.py
python bioelm_deeplinux_1.1.0.py

# 弹性深度分支（1.1.x；顶部 LOOP_* 控制区：K_MAX/训练模式/深度先验/一致性 λ/推理预算 K）
python bioelm_elastic_1.1.1.py
python bioelm_elasticlinux_1.1.1.py
```

仓库根目录**已自带**训练所需文件（`tokenizer_v63b4000.*` 与 `dsr1_v5_0001.jsonl`），
克隆后直接在根目录运行即可。启动后进入 11 项主菜单：

```
1. 训练模型（继续训练）        2. 对话模式
3. 保存模型 / 4. 加载模型      5. 调整生成参数
6. 查看模型统计信息            7. 规则验证拒绝采样：生成正确回答 SFT 数据
8. 基础规则奖励 RL：单回答试错训练   9. GRPO 规则强化学习：同题分组相对优势
10. 退出                       11. 数据预处理：超长对话切分
```

### 关键参数（每个文件顶部「统一控制区」集中管理）

| 参数 | 默认 | 说明 |
|---|---|---|
| `STREAMS_CUDA` | `512` | 训练并行流数（喂饱 GPU 的主旋钮；质量按 512 标定） |
| `BIO_H / BIO_D / BIO_M` | `256/192/192` | 状态单元 / 感官维度 / 读出瓶颈（1.1.6 容量最优比例；[O12] 后换尺寸无需再调 η） |
| `ADAPT_RATE` | `True` | [O11] 自适应学习率：步长 = `s·ρ·R`（R=全体参数 RMS 自派生；s∈[1/8,1] 由损失反馈保守调速） |
| `TRUST_RHO` | `5e-3` | [O11] 无量纲信赖域半径（= 已标定 η/R 口径；换规模不改） |
| `LEARNING_RATE` | `2e-3` | 基准学习率（接口/回退用；`ADAPT_RATE=True` 时无需手调，`False` 回退固定步长） |
| `GRAPH_STEP` | `True` | 训练步 CUDA Graph 整步回放（免 Python 派发；与 eager 逐位等价，失败自动回退） |
| `TRACE_FP16` | `True` | 资格迹 fp16 存储（fp32 归约；带宽减半，长程 A/B 无损；False 回退全 fp32） |
| `GRAD_EINSUM16` | `True` | 迹梯度 fp16 einsum 归约（4.2× 提速；关则回退 fp32 累加路径） |
| `EVAL_STREAMS` | `128` | 评估并行流数（吞吐约 16×，结果口径不变） |
| `EMPTY_CACHE_INTERVAL` | `500` | CUDA 缓存清理间隔（步） |

---

## 目录结构

```
BioELM/
├── bioelm_elite_1.1.6.py        # 单文件产品（通用版；容量最优比例 h=256/d=192/m=192）
├── bioelm_elitelinux_1.1.6.py   # 单文件产品（Linux/WSL 加速版）
├── bioelm_deep_1.1.0.py         # 深度分支产品（[D1] 循环深度 + [D2] 子步时间条件）
├── bioelm_deeplinux_1.1.0.py    # 深度分支产品（[D1]+[D2]，Linux/WSL 加速版）
├── bioelm_elastic_1.1.1.py      # 弹性深度分支产品（[E1] 弹性预算 + [E2] 深度先验）
├── bioelm_elasticlinux_1.1.1.py # 弹性深度分支产品（Linux/WSL 加速版）
├── tokenizer_v63b4000.bin       # 词表（tokenizers 格式）
├── tokenizer_v63b4000_tokenizer.json
├── dsr1_v5_0001.jsonl           # 对话 QA 训练语料（3000 条）
├── experiments/                 # 研究/复现代码（benchmark 台、对手模型、数据构建）
│   ├── qa_data.py               #   从语料构建 qa 数据集（切分/去重/泄露检查/掩码）
│   ├── qa_lm.py                 #   BioELM vs TinyGPT2 同墙钟对决
│   ├── biolssm.py               #   实验版内核
│   ├── gtf2.py                  #   TinyGPT2 对手（标准反传）
│   ├── deep_bench.py            #   深度对照台（循环深度/层堆叠/反馈对齐消融）
│   ├── bench_elastic.py         #   弹性深度台（固定K/随机K/一致性 anytime 评估，小尺度）
│   ├── bench_elastic_gpu.py     #   弹性深度 GPU 台（模式对照：每步质量与每步成本）
│   └── gen_qa.py                #   生成采样
└── bio_lm_v1/data.py            # 实验依赖（词表词库工具）
```

---

## 实验复现

```bash
# 1) 构建数据集（生成 experiments/datasets/qa_*.bin）
python experiments/qa_data.py

# 2) CPU 300s 对决（复现 36.69 vs 61.05）
python experiments/qa_lm.py --budget 300 --only both

# 3) GPU 240s（512 流）
python experiments/qa_lm.py --budget 240 --h 512 --d 128 --m 128 \
    --eta 2e-3 --streams 512 --only bio
```

> **噪声地板提示**：同一命令跨进程重复运行有 ±0.006~0.012 CE 的算法选择噪声
> （cuBLAS heuristics 随内存状态波动）。跨进程对比差异 ≤0.01 不可判定。

---

## 版本历史

### elite / elitelinux 主线（1.1.x）

| 版本 | 关键变化 | 训练吞吐 | 备注 |
|---|---|---|---|
| **1.1.0** | 内核初版：O1–O4（去同步 / foreach 元 Adam / 评估免迹 / NLL 复用）+ 完整 11 项菜单外壳 + RL/GRPO 局部规则化 | 214 步/秒 | 菜单训练冒烟 ppl 18.97 |
| **1.1.1** | [O5] 三条 (B,H,d) 资格迹融合为单张量 `eK(3,B,H,d)`：迹更新 6 算子 → 2 算子，数值等价（max｜Δloss｜=4.8e-7） | 282 步/秒（+32%） | 菜单训练冒烟 best ppl 19.00 |
| **1.1.2** | [O7] 资格迹 `eK` 改 **fp16 存储 + fp32 累加归约**（profiler 定位迹更新占 GPU 时间 86%、带宽受限） | 323 步/秒（受控 +21~24%） | 240s/600s 质量对照无损（15.67/15.38 vs 15.88/15.45） |
| **1.1.3** | [O8a] fp16 einsum 迹梯度 + [O8b] 批量 L1 范数 + [O9] **训练步 CUDA Graph**（同进程逐位等价三重证明，含 90k 步分块形态）+ [O10] 评估流数 8→128 | **569 步/秒**（291k tok/s） | 端到端一轮 6590 步 13.8s（477 步/秒）；评估 92k tok/s |
| **1.1.4** | **自校准内核（消除尺度相关魔数）**：[O11] 自适应学习率（步长=s·ρ·R；损失反馈保守调速，s∈[1/8,1] 结构上不可能爆炸）+ [O12] 派生初始化（记忆半衰期 √H 替代 b_d=3.0）+ [O13] RL/GRPO 优势滚动 RMS 归一化 + [O14] 设备自适应精度 | 与 1.1.3 同档（自适应开销 GPU 上可忽略） | CPU 三尺度免调参验证：h=192 4.595 / h=384 4.619 / h=96 4.588（各尺度手调最优 ±0.04 内）；三版废弃控制器设计留档 |
| **1.1.5** | **[O11-修] 步长基准 R 冻结于建立时（载入刷新）**：修复"权重漂移→R 变大→步长变大"正反馈（弹性一致性训练下实测失控：3000 学习步 R 0.41→1.12、valCE +0.31；冻结后与固定 η 等价） | 同 1.1.4 | h=192 复验 4.611（噪声内）；存载零差 + r0 正确刷新；deep/elastic 线同步修复 |
| **1.1.6** | **容量比例重标定（淘汰赛 Round 11–28）**：`BIO_H 512→256`、`BIO_D 128→192`、`BIO_M 128→192`。读出宽度/输入宽度是真实容量轴且近可加；H>256 饱和、H=512 在 1M 参数档过度；深度（K>1）在新容量点变净负；η=2e-3 与 TAU=1 不变 | 受控 eager A/B 与 1.1.5 持平（约 250 步/秒）；状态迹内存 −25% | 240s 全量 val：16.07 → **15.77**；固定 20k 步：17.31 → **17.01**；1024 流 40k 复验 17.81 → 17.71（打平）且快 25%；长上下文 1024 无反转 |

### deep / deeplinux 分支（1.x）—— 循环深度（回答"单层没潜力 / 多层必须 BP"）

| 版本 | 关键变化 | 验证结果（CPU 500 条同协议，η=2e-3，h=192） | 备注 |
|---|---|---|---|
| **1.0.0** | [D1] **循环深度**：`LOOP_K` 次状态积分后再读出/学习（权重共享、零新增参数；RTRL 原生精确覆盖）；内核与 elite 1.1.4 逐字兼容（`LOOP_K=1` 逐位回退） | 块消融 K=1/2/3/4 → **4.599 / 4.453 / 4.369 / 4.299**；6000 步 K=2: 4.245 vs K=1: 4.440；对照：层监督堆叠 L2 4.713 / L3 5.039，FA 回投 L2 5.659；K=2 在 CPU 上开销 ≈ +13% | deep↔deeplinux 逐位一致；生成/存载/RL 冒烟通过 |
| **1.0.1** | [O11-修] 同步主线修复（R 冻结）；其余与 1.0.0 逐字兼容 | K=2 复验 4.457（1.0.0 为 4.485） | deep↔deeplinux 逐位一致 |
| **1.1.0** | [D2] **子步时间条件注入**：每子步门控偏置 gb[k]/db[k]（零初始化，与 b_g/b_d 同通道精确迹）；**仅 K≥3 启用**（K≤2 自动关闭、逐位退回 1.0.x） | GPU h=512 20k 步：K=3 2.8252→2.8020、K=4 2.8520→2.8012；CPU K=4：3k 步 4.4678→4.3642、10k 步 3.8278→3.7785 | K=2 中性+~6% 开销 → 关闭；逆转深度退化；deep↔deeplinux 逐位一致，K=4 存载零差 |

深度的答案（本期实测）：**堆参数不行的，堆时间可以**——同 token 的循环积分让单层电路获得
深度化的信用分配（资格迹积分 K 子步）与输入累积（`s_K = d^K·s + ((1−d^K)/(1−d))·g·u_raw`），
完全不需要反传或近似回投；层堆叠（层监督/FA）在相同预算下均劣于单层。后续：K 调度、
循环+堆叠混合、GPU 吞吐对照（见 `experiments/deep_bench.py` 复现全部对照）。

### elastic / elasticlinux 分支（1.x）—— 弹性深度（零 autograd，一条模型多档预算）

| 版本 | 关键变化 | 验证结果 | 备注 |
|---|---|---|---|
| **1.0.0** | [E1] **弹性深度**：每训练步同时走完整路线（K_MAX 次积分，CE）与捷径路线（M~U{1..K_MAX−1} 次积分，CE + 一致性蒸馏 `½‖sem_s−sg(sem_f)‖²`）。stop-grad 天然成立，一致性梯度恰为三因子局部形式；顶部 `LOOP_K_MAX / LOOP_TRAIN_MODE / LOOP_CONSIST_LAMBDA / LOOP_K` 四旋钮 | CPU 500 条，h=192，3000 步（anytime @K=1..6）：一致性 **4.354 / 4.194 / 4.172 / 4.178 / 4.189 / 4.204**；固定 K=4 基线 5.033 / 4.525 / 4.353 / 4.299 / 4.286 / 4.290；随机K仅CE 4.651 / 4.509 / 4.474 / 4.468 / 4.474 / 4.487 | 小尺度上任何预算全档优于朴素早退；λ∈[1,3] 宽平顶；含 [O11-修] |
| **1.1.0** | **默认训练模式 consist → random（深度随机化）**：GPU 大尺度(20k 步 h=512)下 random 每步与每墙钟全档最优（K1/K2 3.002/2.846 vs consist 3.306/3.184），成本仅 ~1.07×；consist 降级为小尺度可选模式（负结果记录：λ→0 仍差于 random，更新率减半亦无效） | 240s 同墙钟：52.6k 步 → @K=1 ppl 18.4 / @K=2 **15.86** / @K=3 **15.68**（elite 单层同墙钟 15.54） | 一条模型提供从轻到深的整族预算；复现 `bench_elastic_gpu.py` |
| **1.1.1** | [E2] **训练深度先验**：`LOOP_DEPTH_PRIOR` = uniform（默认）/ shallow（P∝2⁻ᵏ，适 K1/K2）/ deep（P∝k，适 K3/K4）——按目标推理预算选档，零额外成本 | h=192 20k 步 @K1..4：uniform 3.301/3.140/3.115/3.140；shallow **3.157/3.105**/3.146/3.205；deep 3.411/3.150/**3.093/3.107**。240s 墙钟 shallow @K1 2.813（41.8k 步）vs uniform 2.910（52.6k 步） | 同均值 bimodal 劣于 uniform（形状非关键，质量点分布才是） |

弹性深度的用法：训练完一个模型后，推理时只需改顶部 `LOOP_K`（1..K_MAX 任意），无需重训——
"一条训练好的网络里住着从 1 层到 K_MAX 层的整族模型"。已验证的负结果：学习型停机（语义收敛=
固定深度平价、熵判据无信号）、子步时间条件（anytime 下伤浅预算）、K 调度曲线（均劣于固定 K_MAX）——
详见 `expr-*` 实验分支。按目标预算选训练深度先验（`LOOP_DEPTH_PRIOR`：uniform/shallow/deep）是零成本旋钮（见 1.1.1 说明与 `expr-depthdist`）。
复现：`experiments/bench_elastic.py`。

最新版本请见 [Releases](https://github.com/X005-0001-0001/BioELM/releases)；每个版本的
Release 页面由 Zenodo 自动归档并分配 DOI。

---

## 引用

如果本研究对你的工作有帮助，请引用对应版本的 Zenodo DOI（见
[Releases](https://github.com/X005-0001-0001/BioELM/releases) 页面或 `CITATION.cff`）。

## 已知限制（诚实披露）

* 生成内容质量仍处早期阶段：局部规则下 3.3M token 语料的现实水平（语法流畅但内容层面偏语料拼贴），
  瓶颈已定位在**数据规模**而非优化器本身。
* 精确长程召回（recall / 解卷积类任务）仍为开放问题。
* 跨进程 ±0.01 级评测噪声为当前实验基础设施的固有水平。
* 1.1.6 的容量结论主要基于 ≤1024 token 评估与 3.3M token 语料；>2k 长程任务未系统验证。

## 许可证

[AGPL-3.0](LICENSE)。训练语料随仓库提供，仅用于研究与复现。
