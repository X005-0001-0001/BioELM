# BioELM 实验台（研究复现代码）

本目录是 BioELM 的研究与基准代码（当前快照）。产品单文件版（仓库根目录
`bioelm_elite_*.py` / `bioelm_elitelinux_*.py`）与这里的实验内核同源；
实验台用于**同墙钟公平对决**、消融与逐位等价验证。

## 文件

| 文件 | 作用 |
|---|---|
| `qa_data.py` | 从 `../dsr1_v5_0001.jsonl` 构建 qa 数据集：记录级 90/10 切分、去重、跨集泄露检查、仅 assistant 掩码，输出 `datasets/qa_*.bin` |
| `qa_lm.py` | 对决主程序：BioELM vs TinyGPT2，同墙钟预算、分段中评 + 全量终评 |
| `biolssm.py` | 实验版内核（BioLSSM / BioGSSM，含 learn-e、trace_fp16、grad_einsum16、CUDA Graph 回放器 `GraphStepRunner` 与快照/还原） |
| `gtf2.py` | 对手 TinyGPT2（标准反传；仅用于对照） |
| `gen_qa.py` | 从检查点生成样例 |
| `deep_bench.py` | 深度对照台（循环深度 [D1] / 层监督堆叠 / 反馈对齐，r1/r2/r3） |
| `bench_elastic.py` | 弹性深度台（固定K / 随机K / 一致性 λ，anytime@K=1..6，r1/r2） |
| `data2.py` + `../bio_lm_v1/data.py` | 数据流工具（`to_streams` 等） |

依赖：`torch`、`tokenizers`（构建数据集用）。

## 复现步骤

```bash
python qa_data.py                 # 构建 datasets/qa_*.bin（约 1 分钟）
python qa_lm.py --budget 300 --only both          # CPU 对决（36.69 vs 61.05）
python qa_lm.py --budget 240 --h 512 --d 128 --m 128 \
    --eta 2e-3 --streams 512 --only bio --trace-fp16 1 --graph 1
```

`qa_lm.py` 主要参数：`--budget`（墙钟秒）、`--h/--d/--m`（内核尺寸）、`--eta`（元学习率）、
`--streams`、`--learn-e/--e-eta`（可学嵌入）、`--trace-fp16`、`--grad-einsum16`、`--graph`、
`--only bio|tf|both`、`--ckpt`、`--log`。

## 协议（与产品口径一致）

* 语料 3000 条，记录级切分（非 token 尾切），`SEED_SPLIT=7`；
* 全记录哈希去重；跨集 user/assistant 哈希零交集；80 字符近重复探针检查；
* 仅 assistant 区间计入损失（掩码训练 / 掩码验证）；
* 对手 TinyGPT2 使用标准反传，同墙钟 / 同设备 / 同数据流；
* 复现抖动（含跨进程算法选择噪声）：**±0.006~0.012 CE**——≤0.01 的差异不可判定。

## 关键实验结果（RTX 5080 16GB，torch 2.11.0+cu128）

### 内核消融（240s 冻结，512 流，h=512/d=128/m=128，val 子集 CE）

| 配置 | 步数 | 子集 CE@240s | 全量终评 ppl |
|---|---|---|---|
| 基线（fp32 迹，旧核） | 67978 | 2.7292 | 15.88 |
| fp16 迹（O7） | 94884 | 2.7196 | **15.67** |
| fp16 迹 + fp16 einsum 梯度（O8a） | 96279 | 2.7081 | 15.46 |
| 可学嵌入 e_eta=2e-4 | 62919 | 2.7216 | 15.75 |
| d=256 冻结 | 39820 | 2.7441 | 15.92 |
| 600s 长训（fp16 迹） | 222384 | 2.6993 | **15.38** |

结论：e_eta=2e-3 过大（早期"FA 负结果"是假象）；d=256 同预算不划算；
~300s 后进入平台（3.3M token ≈ 26 遍）→ 上限在数据规模。

### CUDA Graph（O9）逐位等价证明

同进程内 eager vs 图回放（同数据、同种子、同步数）：

| 测试 | 规模 | max｜Δparam｜ | CE 差 |
|---|---|---|---|
| 连续跑 | 300 步 | 0 | 0（六位小数） |
| 分块+评估边界 | 15k 步（3×5k） | 0 | 0 |
| 复刻真实块形 | 90k 步（3×30k + 中评） | 0 | 0 |

吞吐：569 步/秒（291k tok/s）；整轮端到端 477 步/秒。捕捉要点（已固化进产品）：
状态就地推进、偏差校正常量张量化、捕捉前快照/还原、静态输入缓冲。

### 负结果留档

* 大 einsum 融合（`'bh,kbhd->khd'`）反而 -25%（-25% 吞吐）；
* Windows 无 triton → `torch.compile` 不可用（Linux 分支可选）；
* 流数 2048 微基准 +15% 但同墙钟质量略降 → 产品默认 512。

### 评估流数（O10）

8 流 5.6k tok/s → 128 流 **92k tok/s（16×）**，结果口径不变（全量掩码 CE）。
