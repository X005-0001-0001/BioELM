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

吞吐（RTX 5080，h=512 / d=128 / m=128，约 97 万参数，512 训练流）：
1.1.0 = 214 步/秒（O1–O4）→ 1.1.1 = 282 步/秒（O5）→ **本版 1.1.2 = 323 步/秒（165k tok/s）**
（O7 fp16 资格迹；受控对照 +21~24%，240s/600s 质量对照无损）。

复现口径与实验矩阵见 [`experiments/README.md`](experiments/README.md)。

---

## 架构一句话

```
g = σ(x·Wg + bg)          # 选择性写入门
d = σ(x·Wd + bd)          # 选择性地漏（初始 b_d=3.0，默认近似恒定记忆）
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
python bioelm_elite_1.1.2.py

# Linux / WSL（带 LINUX_* 加速块：分配器 / 线程 / TF32 / 实验性单步编译，均可关）
python bioelm_elitelinux_1.1.2.py
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
| `BIO_H / BIO_D / BIO_M` | `512/128/128` | 状态单元 / 感官维度 / 读出瓶颈（H=768 时建议 η 降到 1e-3） |
| `LEARNING_RATE` | `2e-3` | 局部规则元学习率 η（内核域标定值） |
| `TRACE_FP16` | `True` | 资格迹 fp16 存储（fp32 归约；带宽减半，长程 A/B 无损；False 回退全 fp32） |
| `EMPTY_CACHE_INTERVAL` | `500` | CUDA 缓存清理间隔（步） |

---

## 目录结构

```
BioELM/
├── bioelm_elite_1.1.2.py        # 单文件产品（通用版）
├── bioelm_elitelinux_1.1.2.py   # 单文件产品（Linux/WSL 加速版）
├── tokenizer_v63b4000.bin       # 词表（tokenizers 格式）
├── tokenizer_v63b4000_tokenizer.json
├── dsr1_v5_0001.jsonl           # 对话 QA 训练语料（3000 条）
├── experiments/                 # 研究/复现代码（benchmark 台、对手模型、数据构建）
│   ├── qa_data.py               #   从语料构建 qa 数据集（切分/去重/泄露检查/掩码）
│   ├── qa_lm.py                 #   BioELM vs TinyGPT2 同墙钟对决
│   ├── biolssm.py               #   实验版内核
│   ├── gtf2.py                  #   TinyGPT2 对手（标准反传）
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

| 版本 | 关键变化 | 训练吞吐 | 备注 |
|---|---|---|---|
| **1.1.0** | 内核初版：O1–O4（去同步 / foreach 元 Adam / 评估免迹 / NLL 复用）+ 完整 11 项菜单外壳 + RL/GRPO 局部规则化 | 214 步/秒 | 菜单训练冒烟 ppl 18.97 |
| **1.1.1** | [O5] 三条 (B,H,d) 资格迹融合为单张量 `eK(3,B,H,d)`：迹更新 6 算子 → 2 算子，数值等价（max｜Δloss｜=4.8e-7） | 282 步/秒（+32%） | 菜单训练冒烟 best ppl 19.00 |
| **1.1.2** | [O7] 资格迹 `eK` 改 **fp16 存储 + fp32 累加归约**（profiler 定位迹更新占 GPU 时间 86%、带宽受限） | **323 步/秒**（受控 +21~24%） | 240s/600s 质量对照无损（15.67/15.38 vs 15.88/15.45） |

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

## 许可证

[AGPL-3.0](LICENSE)。训练语料随仓库提供，仅用于研究与复现。
