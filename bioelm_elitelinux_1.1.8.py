# ================================================================================================
# bioelm_elitelinux_1.1.8.py — 生物局部规则语言模型（BioELM）精英 Linux 加速单文件版
# ------------------------------------------------------------------------------------------------
# 项目名：bioelm    分支：elitelinux（Linux/WSL 加速版）    版本：1.1.8（[O16] label smoothing 0.05 + [O15] 权重 EMA）
# 与 elite 版差异仅三处（全在顶部 LINUX_* 块，可关可回退，数学不变）：分配器/线程/TF32/实验性单步编译。
# 菜单、外壳、内核与 elite 版逐字相同。
#
# 血统：外壳（词表流程 / v5 数据渲染 / 流式解码 / 日志 / brain.pt / 训练与对话流程 /
#       Ultra 三件套（拒绝采样·规则RL·GRPO）/ P4 超长切分 / 主菜单 11 项）来自 3120012elite 系列，
#       **菜单结构与功能逻辑逐项保留**；内核整体替换为 BioELM 电路
#       （选择性门控对角 SSM + 精确在线 RTRL + 三因子局部更新，零 autograd）。
#
# 内核替换清单（原版 → 本版）：
#   复值 SSM / MoE / FP8 / AMP / CUDA Graph / WarmupCosine / 冻结层 / 层监控 → 全部移除
#   原版反传训练（loss.backward + AdamW + grad clip）→ 局部规则在线更新（逐列三因子 + 元 Adam）
#   原版 RL/GRPO（log_probs.backward）→ 局部规则策略梯度（加权 CE 形式，ratio/优势/anchor 按原公式算）
#
# 铁律：
#   1. 训练零 autograd：本文件绝无 loss.backward()；更新全部为局部规则张量化实现。
#   2. 无上下文窗口超参数：状态在线推进，记忆长度由 H 与可学习时间常数决定。
#   3. 对手/对照组禁用本文件。
#
# 内核优化（1.1.0）：
#   [O1] 消除逐步 .item() 同步：ATP 记账全程留在 GPU，日志点才取回（原 9 次/步 CPU-GPU 同步）。
#   [O2] 元 Adam 更新改用 torch._foreach_* 融合批处理（9 参数 × ~7 算子 → ~7 次批量调用）。
#   [O3] 评估/生成路径不再维护资格迹（省 5 个 (B,H,d) 逐列更新），推理明显加速。
#   [O4] step 直接返回掩码 NLL 与计数，训练日志/验证复用同一次前向，不再重复算 softmax。
# 内核优化（1.1.1）：
#   [O5] 三条 (B,H,d) 资格迹融合为单张量 eK(3,B,H,d)：迹更新从 6 算子降为 2 算子，
#        与逐迹版数值等价（max|Δloss|=4.8e-7），全尺寸基准 148.6→262.9 步/秒。
# 内核优化（1.1.2）：
#   [O7] 资格迹 eK 改 fp16 存储（fp32 累加归约）：迹更新带宽减半；实测 +39% 吞吐，
#        240s/600s 质量均不降（冻结终评 15.67/15.38 vs fp32 15.88/15.45）。
#   [O7-注] 已评估未采用：CUDA Graph 只 +13%（启动开销非主瓶颈）；大 einsum 融合反而更慢（-25%）。
# 内核优化（1.1.3）：
#   [O8a] 迹梯度归约改 fp16 einsum（79µs vs mul-sum 333µs/条）；[O8b] 批量 L1 范数收 ATP。
#   [O9] 训练步 CUDA Graph 版（GRAPH_STEP=True）：整步回放免 Python 派发/启动开销；
#        同进程内与 eager 逐位等价（90k 步含分块/评估边界 max|Δparam|=0），实测 569 步/秒。
#        注：本分支若开 LINUX_COMPILE_STEP，建议同时关 GRAPH_STEP（编译融合与图回放二选一，
#        未在 Linux 上组合验证）；GRAPH_STEP 捕捉失败会自动回退 eager。
#   [O10] 评估流数 8→128：评估吞吐 5.6k→92k tok/s（16×）；整轮验证 57s→约 4s。
# 内核优化（1.1.4，与 elite 版逐字相同）——自校准内核：消除尺度相关魔数（不按规模/奖励形状手调）：
#   [O11] 自适应学习率（信赖域 + 保守调速器）：步长 = s·ρ·R（R=全体参数含 E 的 RMS 自派生，无量纲）；
#         s∈[1/8,1] 只保守回退（窗口损失劣于慢迹超噪声时 ÷1.05，恢复 ×1.05 封顶 1），
#         窗口 τ 由电路自身记忆半衰期在线导出，判定阈值由窗口标准误自校准——
#         全程张量运算、无逐步同步、指数域有界（结构上不可能爆炸）。
#   [O12] 派生初始化：记忆半衰期初值 τ0=√H 步（d0=1−ln2/τ0），消除 b_d=3.0 魔数。
#   [O13] RL/GRPO 优势按滚动 RMS 归一化 + 复用自适应步长：不再需要按奖励形状手调 RL 学习率。
#   [O14] 设备自适应精度：fp16 迹/einsum 仅在 CUDA 生效（CPU 自动 fp32）。
#   [O11-修] 步长基准 R 冻结于建立时（载入 brain.pt 时刷新）：逐步实测 R 会随权重漂移形成正反馈
#            （实测 3000 学习步 R 0.41→1.12、6000 步 →2.04），冻结后与固定 η 等价稳定。
# 1.1.4 验证（CPU，dsr1 前 500 条子集，val 掩码 CE，同一份配置免调参；均无 NaN/发散）：
#   h=96: 4.588（旧默认 η=2e-3 为 4.707）｜h=192: 4.595（2e-3: 4.625, 3e-3: 4.550, 1e-2: 4.885）
#   h=384: 4.619（1e-3: 4.685, 2e-3: 4.578, 3e-3: 4.587）｜6000 步 h=192: 4.418（2e-3: 4.475）
#   s 全程 ∈[1/8,1]（基准 = ρ·R ≈ 2.05e-3，即 1.1.0-1.1.3 标定的 η 域）；
#   故意错 4× 的 ρ 仅部分抑制（安全阀，非自动调参器）。
# 已废弃设计留档：[O11-试验1] 逐张量相对尺度（b_d 相对位移放大 ~20×，门控崩溃）；
#   [O11-试验2] 双向升降加速控制器（趋势跟随棘轮，1.0→3.07 早期过冲）；
#   [O11-试验3] 非对称增益（强降弱升，长期下漂 1.0→0.2）。
#
# 实测参考（RTX5080，dsr1_v5_0001.jsonl 掩码 val ppl）：h=512,d=128,m=128 GPU 240s → ppl≈16.2
# ================================================================================================

# ============================================================
# 统一控制区（所有超参数/开关集中在此）
# ============================================================
# ---- 原版外壳常量（保留原名与原语义；★=因内核替换已标定新值） ----
VOCAB_SIZE = 4000                # 训练新词表时的目标词表（原版 10000；本仓库既有词表为 4000 档）
LEARNING_RATE = 2.0e-3           # ★基准学习率（接口/回退用；[O11] 自适应开启时不再需要按规模手调）
WEIGHT_DECAY = 1.0e-4            # ★= 内核 wd（仅二维参数）
GRAD_CLIP_NORM = 1.0             # 保留接口（局部规则无梯度裁剪；训练开始时会提示）
BATCH_SIZE = 1                   # 保留接口（内核为在线逐列更新，不使用批大小）
EPOCHS = 1                       # 训练轮数（每轮=对训练语料完整过一遍流）
LOG_INTERVAL = 2000              # 每多少步打一条训练日志（0=不打中途日志）
TEMPERATURE = 0.7                # 生成温度（菜单 5 可调）
TOP_P = 0.9                      # 生成 top-p（菜单 5 可调）
GRADIENT_ACCUMULATION_STEPS = 16 # 保留接口（内核无梯度累积）
MAX_SEQ_LEN = 4096               # 对话历史窗口 / 生成上限（内核训练流长度另行计算）
EARLY_STOP_PATIENCE = 3          # 验证不改善的耐心轮数
BPE_MIN_FREQUENCY = 2            # 训练新词表时的最小词频
NORMALIZE_MODE = 'none'          # 'none' = 原样（保空格/换行/大小写/全角标点）
TRAIN_ON_REASONING = True        # assistant 的 reasoning_content 是否纳入训练范围
RL_PROMPT_SUFFIX = "\n\n请自由推理。最后必须单独写一行：最终答案：你的答案"
RL_SYSTEM_PROMPT = ""            # 非空时作为 system 段渲染（agent/固定指令用）
P4_SEG_MAX_TOKENS = 2048         # P4 切分：单样本 token 上限

# ---- Ultra 三件套默认参数（菜单 7/8/9；★=内核标定值） ----
ULTRA_RS_SAMPLES_PER_QUESTION = 8
ULTRA_RS_MAX_NEW_TOKENS = 512
ULTRA_RL_EPOCHS = 1
ULTRA_RL_MAX_NEW_TOKENS = 512
ULTRA_RL_LEARNING_RATE = 2.0e-4    # ★局部规则标定（原版反传 AdamW 为 1e-6）
ULTRA_RL_BASELINE_MOMENTUM = 0.9
ULTRA_GRPO_EPOCHS = 1
ULTRA_GRPO_GROUP_SIZE = 8
ULTRA_GRPO_MAX_NEW_TOKENS = 512
ULTRA_GRPO_LEARNING_RATE = 2.0e-4  # ★同上
ULTRA_GRPO_INNER_STEPS = 1
ULTRA_GRPO_CLIP_EPSILON = 0.2
ULTRA_GRPO_OLD_POLICY_PENALTY = 0.01

# ---- BioELM 内核 ----
BIO_H = 256                      # 状态单元数 H（淘汰赛 Round 19/20/21：H>256 饱和，512 过度）
BIO_D = 192                      # 有界嵌入维度 d（Round 12/15/17：d 是最高效容量轴）
BIO_M = 192                      # 读出瓶颈 sem 维度 m（Round 11/13：M≈H 为性价比拐点）
BIO_BETA1 = 0.9                  # 元动量一阶
BIO_BETA2 = 0.99                 # 元动量二阶
BIO_EPS = 1e-6                   # 元归一化防零
BIO_SEED = 43                    # 初始化种子
STREAMS_CUDA = 512               # GPU 并行流数（喂饱 GPU 的关键）
STREAMS_CPU = 64                 # CPU 并行流数
VAL_FRAC = 0.10                  # 记录级验证集占比（防泄露，不用 token 尾切）
SEED_SPLIT = 7                   # 数据切分种子
DEDUP_RECORDS = True             # 全记录哈希去重
LEAK_PROBE_CHARS = 80            # 跨集近重复探测前缀长度（0=关闭）
EMPTY_CACHE_INTERVAL = 500       # cuda 缓存清理间隔（步；0=关闭）
TRACE_FP16 = True                # [O7] 资格迹 fp16 存储（fp32 归约）；False=全 fp32（对照用）
GRAD_EINSUM16 = True             # [O8a] 迹梯度 fp16 einsum 归约；False=mul-sum（fp32 累加，慢 4×）
GRAPH_STEP = True                # [O9] 训练步 CUDA Graph 回放（免 Python 派发）；捕捉失败自动回退 eager
EMA_DECAY = 0.999                # [O15] 权重指数平均（Polyak/EMA）：训练期维护，评估/保存/部署用平均权重（推理零额外成本）
BIO_LSM = 0.05                   # [O16] label smoothing（训练目标平滑；评估仍用硬目标）
EVAL_STREAMS = 128               # [O10] 评估并行流数（8→128：评估吞吐约 16×，结果口径不变）

# ---- 自校准内核（O11–O13）：全部无量纲/自派生，换规模换奖励不改配置 ----
ADAPT_RATE = True                # [O11] 自适应学习率（损失反馈+信赖域）；False=回退 LEARNING_RATE 固定步长
TRUST_RHO = 5.0e-3               # [O11] 信赖域半径（无量纲）：步长 = s·ρ·R，R=全体参数（含冻结嵌入 E）RMS
DIR_CLAMP = 8.0                  # [O11] 元 Adam 方向逐元素钳位（方向 RMS≈1，±8=离群防护）
CTRL_KAPPA = 0.05                # [O11] 调速器每窗乘性调节量（对数域 5%）
CTRL_S_MIN = 2.0 ** -3           # [O11] 调速器下界（1/8）
CTRL_S_MAX = 1.0                 # [O11] 上界 = 基准（ρ·R ≈ 已标定 η；只保守回退、不激进超调）
RETENTION_TAU0 = None            # [O12] 记忆半衰期初值/步（None=自动 √H）；d0=1−ln2/τ0，b_d=logit(d0)

# ---- 路径 / 设备 ----
TOKENIZER_FILES = ['tokenizer_incremental.bin', 'tokenizer_v63b4000.bin']  # 依次尝试加载
LOG_FILE = 'training_log.txt'
DEVICE_PREF = 'auto'             # 'auto' | 'cpu' | 'cuda'（BIO_DEV 环境变量优先）

# ============================================================
# LINUX_* 加速块（仅本文件；全部可关、失败自动回退，数学不变）
#   说明：Win11 + torch 2.11 下 expandable_segments 会触发分配器 invalid argument
#   （平台限制），故仅 Linux 生效；Windows 上运行本文件会自动跳过该设置。
# ============================================================
LINUX_EXPANDABLE_SEGMENTS = True   # Linux 上设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LINUX_NUM_THREADS = 0              # CPU 线程数；0=自动取满核
LINUX_USE_TF32 = False             # True=允许 matmul/cudnn TF32（略提速、精度降；默认关）
LINUX_COMPILE_STEP = False         # 实验性：尝试对内核单步 torch.compile（失败/异常静默回退 eager）

import os as _os_env, sys as _sys_env
if LINUX_EXPANDABLE_SEGMENTS and _sys_env.platform.startswith('linux'):
    _os_env.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import contextlib
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation
from fractions import Fraction

import torch

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False

# -------- 环境变量覆盖（训练期超参） --------
EMA_DECAY = float(os.environ.get('BIO_EMA', EMA_DECAY))   # BIO_EMA=0 关闭 EMA
BIO_LSM = float(os.environ.get('BIO_LSM', BIO_LSM))       # BIO_LSM=0 关闭平滑

# -------- 设备选择（BIO_DEV 环境变量可覆盖：cpu/cuda） --------
_dev_env = os.environ.get('BIO_DEV', '').strip().lower()
if _dev_env in ('cpu', 'cuda'):
    DEV = torch.device(_dev_env)
elif DEVICE_PREF == 'cuda' or (DEVICE_PREF == 'auto' and torch.cuda.is_available()):
    DEV = torch.device('cuda')
else:
    DEV = torch.device('cpu')
if LINUX_USE_TF32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(LINUX_NUM_THREADS if LINUX_NUM_THREADS > 0 else (os.cpu_count() or 8))


# ============================================================
# 日志（与原版一致：写 training_log.txt）
# ============================================================
_LOG_FILE = None
_LOG_FLUSH_COUNTER = 0
_LOG_FLUSH_INTERVAL = 50


def init_log_file():
    global _LOG_FILE
    if _LOG_FILE is None:
        _LOG_FILE = open(LOG_FILE, 'a', encoding='utf-8')
        _LOG_FILE.write('\n--- 新会话开始于 %s ---\n' % datetime.now())
        _LOG_FILE.flush()
        log_and_print('设备: %s | 内核: BioELM（选择性门控 SSM + 精确在线 RTRL + 三因子局部更新，零 autograd）' % DEV)


def log_and_print(*args, **kwargs):
    global _LOG_FLUSH_COUNTER
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    msg = sep.join(str(arg) for arg in args) + end
    sys.stdout.write(msg)
    sys.stdout.flush()
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg)
        _LOG_FLUSH_COUNTER += 1
        if _LOG_FLUSH_COUNTER >= _LOG_FLUSH_INTERVAL:
            _LOG_FILE.flush()
            _LOG_FLUSH_COUNTER = 0


def close_log_file():
    global _LOG_FILE
    if _LOG_FILE is not None:
        _LOG_FILE.flush()
        _LOG_FILE.close()
        _LOG_FILE = None


# ============================================================
# 工具函数 / v5 数据格式层（原样保留）
# ============================================================
def _active_normalize_mode():
    return NORMALIZE_MODE


def normalize_text(text: str) -> str:
    if _active_normalize_mode() == 'nfkc':
        return unicodedata.normalize('NFKC', text)
    return text


def _messages_from_record(rec):
    """取出一行数据的 messages（v5 唯一格式；旧 conversations 不再兼容，遇到就报错）。"""
    if isinstance(rec, list):
        msgs = rec
    else:
        msgs = rec.get('messages')
        if msgs is None:
            raise ValueError("数据缺少 messages 字段：v5 不保留旧 conversations 兼容，"
                             "请先用 _to_v5.py 把语料/题库转成 {'messages': [...]}")
    for _m in msgs:
        _r = _m.get('role')
        if _r not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('非法 role %r：只允许 system/user/assistant/tool' % (_r,))
        if _m.get('content') is None and not _m.get('tool_calls'):
            raise ValueError('消息既没有 content 也没有 tool_calls')
    return [dict(m) for m in msgs]


def _render_tools(tools):
    if not tools:
        return ''
    return '<|user|>[工具]' + json.dumps(tools, ensure_ascii=False) + '<|end|>'


def _render_tool_calls(tcs):
    if not tcs:
        return ''
    return json.dumps(tcs, ensure_ascii=False)


def render_messages(messages, tools=None, add_generation_prompt=False):
    """渲染 messages → (text, spans)；spans 为需要计算 loss 的字符区间（assistant 段）。"""
    parts = []
    spans = []
    pos = 0

    def emit(s, train=False):
        nonlocal pos
        n = normalize_text(s)
        if train:
            spans.append((pos, pos + len(n)))
        parts.append(n)
        pos += len(n)

    emit('<|bos|>')
    if tools:
        emit(_render_tools(tools))
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content')
        content = '' if content is None else str(content)
        if role == 'system':
            emit('<|user|>[系统]' + content + '<|end|>')
        elif role == 'user':
            emit('<|user|>' + content + '<|end|>')
        elif role == 'tool':
            tid = msg.get('tool_call_id')
            emit('<|user|>[工具结果' + ((' id=' + str(tid)) if tid else '') + ']' + content + '<|end|>')
        elif role == 'assistant':
            emit('<|assistant|>')
            seg = []
            rc = msg.get('reasoning_content')
            if TRAIN_ON_REASONING and rc:
                seg.append(str(rc))
            seg.append(content)
            tc = _render_tool_calls(msg.get('tool_calls'))
            if tc:
                seg.append(tc)
            emit(''.join(seg) + '<|end|>', train=True)
    if add_generation_prompt:
        emit('<|assistant|>')
    else:
        emit('<|eos|>', train=True)
    return ''.join(parts), spans


def _split_reasoning(text: str):
    """拆出 thinking 段与正文；无标签 → 整段算正文（v5 预期路径）。"""
    if not text:
        return '', ''
    for _open, _close in (
        ('<' + 'think' + '>', '<' + '/think' + '>'),
        ('<' + 'thinking' + '>', '<' + '/thinking' + '>'),
    ):
        m = re.search(re.escape(_open) + r'(.*?)' + re.escape(_close), text, flags=re.S)
        if m:
            reasoning = m.group(1).strip()
            body = (text[:m.start()] + text[m.end():]).strip()
            return reasoning, body
    return '', text.strip()


# ============================================================
# 流式安全解码（原样保留）
# ============================================================
class StreamDecoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.buf = []
        self.emitted = ''

    def _stable(self):
        cur = self.tokenizer.decode(self.buf)
        if cur.endswith('\ufffd'):
            cur = cur[:-1]
        return cur

    def push(self, token_id):
        self.buf.append(token_id)
        cur = self._stable()
        if not cur.startswith(self.emitted):
            return ''
        out = cur[len(self.emitted):]
        self.emitted = cur
        return out

    def flush(self):
        cur = self.tokenizer.decode(self.buf)
        out = cur[len(self.emitted):] if cur.startswith(self.emitted) else ''
        self.emitted = cur
        return out


# ============================================================
# 词表（ByteLevel BPE；主文件 <base>_tokenizer.json，命名沿用原版）
# ============================================================
SPECIAL_TOKENS_LIST = ['<|bos|>', '<|eos|>', '<|pad|>', '<|unk|>', '<|user|>', '<|assistant|>', '<|end|>']


class Vocab:
    def __init__(self):
        self._tokenizer = None
        self.ids = {}
        self._vocab_size_cache = 0

    # ---- 原版兼容属性 ----
    @property
    def vocab_size(self):
        return self._vocab_size_cache

    @property
    def bos_token_id(self):
        return self.ids.get('<|bos|>')

    @property
    def eos_token_id(self):
        return self.ids.get('<|eos|>')

    @property
    def pad_token_id(self):
        return self.ids.get('<|pad|>')

    @property
    def unk_token_id(self):
        return self.ids.get('<|unk|>')

    @property
    def user_token_id(self):
        return self.ids.get('<|user|>')

    @property
    def assistant_token_id(self):
        return self.ids.get('<|assistant|>')

    @property
    def end_token_id(self):
        return self.ids.get('<|end|>')

    def train(self, texts, vocab_size=VOCAB_SIZE, min_freq=BPE_MIN_FREQUENCY, verbose=False):
        tk = Tokenizer(models.BPE(unk_token='<|unk|>'))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tr = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS_LIST,
                                 min_frequency=min_freq, show_progress=verbose,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tk.train_from_iterator(list(texts), trainer=tr)
        tk.decoder = decoders.ByteLevel()
        self._tokenizer = tk
        self._refresh()

    def load(self, path):
        base = os.path.splitext(path)[0]
        json_path = base + '_tokenizer.json'
        if path.endswith('_tokenizer.json'):
            json_path = path
        if not os.path.exists(json_path):
            raise FileNotFoundError('词表文件缺失: ' + json_path)
        tk = Tokenizer.from_file(json_path)
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tk.decoder = decoders.ByteLevel()
        self._tokenizer = tk
        self._refresh()
        return True

    def _refresh(self):
        v = self._tokenizer.get_vocab()
        self.ids = {t: v.get(t, i) for i, t in enumerate(SPECIAL_TOKENS_LIST)}
        self._vocab_size_cache = self._tokenizer.get_vocab_size()

    def encode(self, text):
        return list(self._tokenizer.encode(text).ids)

    def decode(self, ids):
        return self._tokenizer.decode(list(ids), skip_special_tokens=True)

    def save(self, path):
        base = os.path.splitext(path)[0]
        self._tokenizer.save(base + '_tokenizer.json')
        log_and_print('词表已保存: %s_tokenizer.json（词表大小 %d）' % (base, self.vocab_size))


def _encode_with_offsets(tokenizer, text: str):
    if tokenizer._tokenizer is None:
        raise RuntimeError('Tokenizer 未训练或未加载')
    enc = tokenizer._tokenizer.encode(text)
    ids = [tid if tid < tokenizer._vocab_size_cache else tokenizer.unk_token_id for tid in enc.ids]
    return ids, list(enc.offsets)


def _encode_with_mask(tokenizer, text, spans):
    ids, offsets = _encode_with_offsets(tokenizer, text)
    mask = [0] * len(ids)
    if spans:
        for i, (s, e) in enumerate(offsets):
            for a, b in spans:
                if s >= a and s < b:
                    mask[i] = 1
                    break
    return ids, mask


# ============================================================
# 语料装载：去重 / 记录级切分 / 泄露检查 / 拉平 + 掩码
# ============================================================
def _norm_hash(s):
    return hashlib.sha1(s.strip().encode('utf-8')).hexdigest()


def build_corpus(tokenizer, corpus_path):
    recs = []
    skipped = 0
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                msgs = _messages_from_record(rec)
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    log_and_print('跳过无效行: %s' % e)
                continue
            if not any(m.get('role') == 'assistant' for m in msgs):
                skipped += 1
                continue
            recs.append({'messages': msgs})
    n_raw = len(recs) + skipped
    if DEDUP_RECORDS:
        seen = set()
        uniq = []
        for r in recs:
            h = _norm_hash(json.dumps(r, ensure_ascii=False, sort_keys=True))
            if h not in seen:
                seen.add(h)
                uniq.append(r)
        recs = uniq
    log_and_print('[DATA] records=%d(可用) 无效/跳过=%d 去重后=%d' % (n_raw, skipped, len(recs)))

    rng = random.Random(SEED_SPLIT)
    order = list(range(len(recs)))
    rng.shuffle(order)
    n_val = max(1, int(len(recs) * VAL_FRAC))
    val_idx = set(order[:n_val])
    train_recs = [recs[i] for i in order if i not in val_idx]
    val_recs = [recs[i] for i in order[:n_val]]

    # 泄漏检查（哈希 + 前缀探针），与原内核流水线一致
    tr_user, tr_asst = set(), set()
    for r in train_recs:
        for m in r['messages']:
            h = _norm_hash(m.get('content') or '')
            (tr_user if m.get('role') == 'user' else tr_asst).add(h)
    leak_u = leak_a = 0
    keep_val = []
    for r in val_recs:
        bad = False
        for m in r['messages']:
            h = _norm_hash(m.get('content') or '')
            if m.get('role') == 'user' and h in tr_user:
                leak_u += 1
                bad = True
            if m.get('role') == 'assistant' and h in tr_asst:
                leak_a += 1
                bad = True
        if not bad:
            keep_val.append(r)
    log_and_print('[DATA] 泄露检查(哈希): val user 重合=%d assistant 重合=%d 保留=%d'
                  % (leak_u, leak_a, len(keep_val)))
    if LEAK_PROBE_CHARS > 0:
        blob_u = '\n'.join((m.get('content') or '').strip() for r in train_recs
                           for m in r['messages'] if m.get('role') == 'user')
        blob_a = '\n'.join((m.get('content') or '').strip() for r in train_recs
                           for m in r['messages'] if m.get('role') == 'assistant')
        su = sa = 0
        for r in keep_val:
            for m in r['messages']:
                probe = (m.get('content') or '').strip()[:LEAK_PROBE_CHARS]
                if not probe:
                    continue
                if m.get('role') == 'user' and probe in blob_u:
                    su += 1
                if m.get('role') == 'assistant' and probe in blob_a:
                    sa += 1
        log_and_print('[DATA] 近重复探针(%d字符): user 命中=%d assistant 命中=%d（需人工关注套话/模板）'
                      % (LEAK_PROBE_CHARS, su, sa))

    def build(records):
        all_ids, all_mask = [], []
        for r in records:
            text, spans = render_messages(r['messages'])
            ids, msk = _encode_with_mask(tokenizer, text, spans)
            all_ids.extend(ids)
            all_mask.extend(msk)
        return all_ids, all_mask

    tr_ids, tr_mask = build(train_recs)
    va_ids, va_mask = build(keep_val)
    log_and_print('[DATA] train=%d(learn %d) val=%d(learn %d)'
                  % (len(tr_ids), sum(tr_mask), len(va_ids), sum(va_mask)))
    return tr_ids, tr_mask, va_ids, va_mask


def _shift_next(ids):
    return ids[1:] + ids[:1]


def to_streams(ids, mask, n_streams, rotate=0):
    """把扁平 token 序列切成 n_streams 条平行流；rotate 用于每轮轻微换样。"""
    n = len(ids)
    if rotate:
        ids = ids[rotate:] + ids[:rotate]
        mask = mask[rotate:] + mask[:rotate]
    T = n // n_streams
    if T < 1:
        raise ValueError('语料 token 太少，无法切成 %d 条流' % n_streams)
    cut = n - T * n_streams
    ids = ids[cut:]
    mask = mask[cut:]
    st = torch.tensor(ids, dtype=torch.long).reshape(n_streams, T)
    mk = torch.tensor(mask, dtype=torch.float).reshape(n_streams, T)
    tg = torch.tensor(_shift_next(ids), dtype=torch.long).reshape(n_streams, T)
    return st, tg, mk


# ============================================================
# 内核：BioELM（选择性门控对角 SSM，精确在线 RTRL，零 autograd）
# ------------------------------------------------------------
#   g = sigmoid(x·Wg+bg)          写入门
#   d = sigmoid(x·Wd+bd)          选择性遗忘（默认记住，b_d 初始 3.0）
#   s = d⊙s_prev + g⊙(x·B)        状态（对角线 -> 资格迹逐列精确）
#   sem = sigmoid(s·Wp+bp) -> logits = sem·head^T + bout
#   更新：对偶学习信号 e 沿资格迹做局部三因子乘法 -> 元 Adam（RMS 归一）
#   [O1]-[O4] 见文件头优化清单
# ============================================================
def _foreach_ok():
    try:
        a = [torch.zeros(2), torch.ones(3)]
        b = [torch.ones(2), torch.ones(3)]
        torch._foreach_mul_(a, 0.9)
        torch._foreach_add_(a, b, alpha=0.1)
        torch._foreach_addcmul_(a, b, b, value=0.5)
        m = torch._foreach_mul(a, 2.0)
        v = torch._foreach_sqrt(torch._foreach_div(a, 2.0))
        torch._foreach_add_(v, 1e-6)
        u = torch._foreach_div(m, v)
        torch._foreach_add_(a, u, alpha=-0.1)
        return True
    except Exception:
        return False


_FOREACH_OK = _foreach_ok()


class BioELM:
    def __init__(self, vocab_size, d=BIO_D, h=BIO_H, m=BIO_M, seed=BIO_SEED):
        g = torch.Generator().manual_seed(seed)
        self.vocab = vocab_size
        self.d, self.h, self.m = d, h, m
        self.E = (torch.randn(vocab_size, d, generator=g) * 0.6).to(DEV)
        self.W_g = ((torch.rand(h, d, generator=g) * 2 - 1) / math.sqrt(d)).to(DEV)
        self.b_g = torch.zeros(h).to(DEV)
        self.W_d = ((torch.rand(h, d, generator=g) * 2 - 1) / math.sqrt(d)).to(DEV)
        # [O12] 派生初始化：记忆半衰期 τ0=√H 步（d0=1−ln2/τ0，b_d=logit(d0)）——消除固定魔数 3.0
        tau0 = float(RETENTION_TAU0) if RETENTION_TAU0 else math.sqrt(h)
        d0 = 1.0 - math.log(2.0) / tau0
        self.b_d = torch.full((h,), math.log(d0 / (1.0 - d0))).to(DEV)
        self.B = (torch.randn(h, d, generator=g) * (1.0 / math.sqrt(d))).to(DEV)
        self.Wp = ((torch.rand(m, h, generator=g) * 2 - 1) / math.sqrt(h)).to(DEV)
        self.bp = torch.zeros(m).to(DEV)
        self.head = ((torch.rand(vocab_size, m, generator=g) * 2 - 1) / math.sqrt(m)).to(DEV)
        self.bout = torch.zeros(vocab_size).to(DEV)
        self.eta, self.beta1, self.beta2 = LEARNING_RATE, BIO_BETA1, BIO_BETA2
        self.eps, self.wd = BIO_EPS, WEIGHT_DECAY
        self._meta = {k: (torch.zeros_like(p), torch.zeros_like(p))
                      for k, p in self.params().items()}
        self._ema = None
        if EMA_DECAY > 0.0:
            self._ema = {k: p.detach().clone() for k, p in self.params().items()}
        self.steps = 0
        self.use_foreach = _FOREACH_OK
        self._ar_cache = {}
        # [O9] 偏差校正常量张量化（CUDA Graph 捕捉安全：值由 _set_bias_factors 在回放前写入）
        self._b1c = torch.ones((), device=DEV)
        self._b2c = torch.ones((), device=DEV)
        self._b1c_inv = torch.ones((), device=DEV)
        self._b2c_inv = torch.ones((), device=DEV)
        # [O14] 设备自适应精度：fp16 仅在 CUDA 生效（CPU 半精度无向量化慢路径，自动回退 fp32）
        self.trace_fp16 = bool(TRACE_FP16) and DEV.type == 'cuda'
        self.grad_einsum16 = bool(GRAD_EINSUM16) and DEV.type == 'cuda'
        # ---- [O11] 自适应学习率状态（持久张量缓冲：原地更新，CUDA Graph 回放时被读取到新值） ----
        self.adapt = bool(ADAPT_RATE)
        self._s = torch.ones((), device=DEV)           # 反馈因子（对数域乘性，∈[CTRL_S_MIN, CTRL_S_MAX]）
        self._loss_ema = torch.zeros((), device=DEV)   # 慢窗损失迹
        self._ema_init = False                         # 首窗标志（tick 在捕捉外执行，可用 Python 分支）
        self._win_sum = torch.zeros((), device=DEV)    # 当前窗口 per-step 损失和（张量，可被图回放推进）
        self._win_sq = torch.zeros((), device=DEV)
        self._win_cnt = 0
        self._win_len = self._derive_window()          # 窗口 = 电路自身记忆半衰期（在线导出）
        self._w_rms2 = torch.ones((), device=DEV)      # [O13] RL/GRPO 优势滚动 RMS²（无量纲化奖励形状）
        # [O11-修] 步长基准 R 冻结于建立时（全体参数含 E 的 RMS）：逐步实测 R 会随权重漂移
        # 形成正反馈（实测 3000 学习步 R 0.41→1.12、6000 步 →2.04，步长自激失控）；载入 brain.pt 时刷新。
        self._r0 = self._compute_r0()

    # ---- 参数表 ----
    def params(self):
        return {'W_g': self.W_g, 'b_g': self.b_g, 'W_d': self.W_d, 'b_d': self.b_d,
                'B': self.B, 'Wp': self.Wp, 'bp': self.bp, 'head': self.head,
                'bout': self.bout}

    def n_params(self):
        return sum(t.numel() for t in self.params().values()) + self.E.numel()


    def _arange(self, B):
        t = self._ar_cache.get(B)
        if t is None:
            t = torch.arange(B, device=DEV)
            self._ar_cache[B] = t
        return t

    # ---- [O9] 图捕捉配套：偏差常量张量化 + 训练状态快照/还原 ----
    def _set_bias_factors(self):
        t = self.steps + 1
        b1c = 1.0 - self.beta1 ** t
        b2c = 1.0 - self.beta2 ** t
        self._b1c.fill_(b1c)
        self._b2c.fill_(b2c)
        self._b1c_inv.fill_(1.0 / b1c)
        self._b2c_inv.fill_(1.0 / b2c)

    # ---- [O11] 自适应学习率：窗口派生 + 反馈控制 ----
    def _compute_r0(self):
        """[O11-修] 步长基准：全体模型参数（含冻结嵌入 E）的 RMS，建立/载入时计算一次。"""
        ps = list(self.params().values())
        tot = sum((p * p).sum() for p in ps) + self.E.square().sum()
        numel = sum(p.numel() for p in ps) + self.E.numel()
        return (tot / numel).clamp(min=1e-12).sqrt().detach()

    def _derive_window(self):
        """反馈窗口 = 电路自身记忆半衰期 τ = ln2/(1−d̄)，d̄ 由当前 b_d 在线导出；钳位 [2^5, 2^12] 步。"""
        d_bar = torch.sigmoid(self.b_d).mean()
        one_m = (1.0 - d_bar).clamp(min=2.0 ** -12)
        return int((math.log(2.0) / one_m).clamp(2.0 ** 5, 2.0 ** 12).item())

    def _controller_tick(self):
        """保守稳态调速器：s∈[1/8,1]，基准=ρ·R（自派生）。
        规则：窗口损失劣于慢迹超过噪声 σ → 乘性回退；优于 σ → 乘性恢复（上限=基准）；
        其余保持。只保守回退、不激进超调——结构上不可能因反馈爆炸。
        （负结果留档：[O11-试验] 双向升降的"加速型"控制器会趋势跟随棘轮上调
          （损失单调改善期每窗 +5% 无上限累计），实测早期过冲致门控退化 valCE 4.55→5.08，已废弃。）"""
        w = float(self._win_cnt)
        l_mean = self._win_sum / w
        var = (self._win_sq / w - l_mean * l_mean).clamp(min=0.0)
        sigma = float((var / w).sqrt()) + 1e-12     # 窗口均值的标准误（噪声自校准）
        if not self._ema_init:
            self._loss_ema.copy_(l_mean)
            self._ema_init = True
        else:
            delta = float(self._loss_ema - l_mean)  # 正 = 比慢迹更好
            r = delta / sigma                       # 以噪声 σ 为单位的偏离量（无量纲）
            # 对数域对称比例控制（死区 ±1σ，饱和 ±3σ）：对称增益在对称噪声下零漂移，
            # 强降弱升的非对称增益会致 s 长期下漂（实测 6000 步漂至 0.2，已修正）
            if r > 1.0:
                gain = math.exp(CTRL_KAPPA * min(r - 1.0, 3.0))    # 优于噪声 → 向基准恢复（封顶 s=1）
            elif r < -1.0:
                gain = math.exp(-CTRL_KAPPA * min(-r - 1.0, 3.0))  # 劣于噪声 → 比例回退（最大 ~16%/窗）
            else:
                gain = 1.0
            self._s.mul_(gain).clamp_(CTRL_S_MIN, CTRL_S_MAX)
            self._loss_ema.mul_(1.0 - 1.0 / 5.0).add_(l_mean, alpha=1.0 / 5.0)   # 慢迹 ≈ 5 窗
        self._win_sum.zero_()
        self._win_sq.zero_()
        self._win_cnt = 0
        self._win_len = self._derive_window()

    def _snapshot_train_state(self):
        return {
            'steps': self.steps,
            'params': {k: v.detach().clone() for k, v in self.params().items()},
            'E': self.E.detach().clone(),
            'meta': {k: (m[0].clone(), m[1].clone()) for k, m in self._meta.items()},
            'ema': ({k: v.detach().clone() for k, v in self._ema.items()}
                    if self._ema is not None else None),
            'ctrl': (self._s.clone(), self._loss_ema.clone(),
                     self._win_sum.clone(), self._win_sq.clone(),
                     self._win_cnt, self._win_len, self._ema_init),
        }

    def _restore_train_state(self, snap):
        self.steps = snap['steps']
        for k, v in self.params().items():
            v.copy_(snap['params'][k])
        self.E.copy_(snap['E'])
        for k, m in self._meta.items():
            m[0].copy_(snap['meta'][k][0])
            m[1].copy_(snap['meta'][k][1])
        if snap.get('ema') is not None and self._ema is not None:
            for k, v in self._ema.items():
                v.copy_(snap['ema'][k])
        self._s.copy_(snap['ctrl'][0])
        self._loss_ema.copy_(snap['ctrl'][1])
        self._win_sum.copy_(snap['ctrl'][2])
        self._win_sq.copy_(snap['ctrl'][3])
        self._win_cnt = snap['ctrl'][4]
        self._win_len = snap['ctrl'][5]
        self._ema_init = snap['ctrl'][6]

    def init_stream_state(self, batch):
        return {
            's': torch.zeros(batch, self.h, device=DEV),
            # [O5] 三条 (B,H,d) 资格迹融合为单张量 eK(3,B,H,d)，首维连续便于切片求梯度
            # [O7]+[O14] 存储精度按设备自适应选择（CUDA→fp16，CPU→fp32）
            'eK': torch.zeros(3, batch, self.h, self.d,
                              dtype=(torch.float16 if self.trace_fp16 else torch.float32),
                              device=DEV),
            'e_bg': torch.zeros(batch, self.h, device=DEV),
            'e_bd': torch.zeros(batch, self.h, device=DEV),
        }

    # ---- 单列前向（trace=False 时不维护资格迹：[O3] 评估/生成提速） ----
    @torch.no_grad()
    def _cell(self, tok, st, trace=True):
        x = self.E[tok]
        g = torch.sigmoid(x @ self.W_g.t() + self.b_g)
        d = torch.sigmoid(x @ self.W_d.t() + self.b_d)
        u_raw = x @ self.B.t()
        u = g * u_raw
        # [O9] 状态就地推进（不重绑定张量 -> CUDA Graph 回放时状态可持续前进）
        s_buf = st['s']
        if trace:
            c_g = (g * (1.0 - g)) * u_raw
            c_d = (d * (1.0 - d)) * s_buf        # 必须先于 s 就地更新计算
        s_buf.mul_(d).add_(u)
        s = s_buf
        sem = torch.sigmoid(s @ self.Wp.t() + self.bp)
        logits = sem @ self.head.t() + self.bout
        if trace:
            # [O5] 一次乘 + 一次 addcmul 完成三条迹更新（原为 6 个算子）
            eK = st['eK']
            if self.trace_fp16:
                # [O7] fp16 存储：更新算子同数学，舍入误差由长程 A/B 对照标定（不降质）
                eK.mul_(d.half().unsqueeze(-1).unsqueeze(0))
                eK.addcmul_(torch.stack((c_g, c_d, g), dim=0).half().unsqueeze(-1),
                            x.half().unsqueeze(0).unsqueeze(2))
            else:
                eK.mul_(d.unsqueeze(-1).unsqueeze(0))
                eK.addcmul_(torch.stack((c_g, c_d, g), dim=0).unsqueeze(-1),
                            x.unsqueeze(0).unsqueeze(2))
            st['e_bg'].mul_(d).add_(c_g)
            st['e_bd'].mul_(d).add_(c_d)
        return logits, sem

    # ---- 局部三因子更新（学习信号 e -> 资格迹 -> 元 Adam） ----
    def _learn(self, sem, s, st, e):
        g_sem = (e @ self.head) * (sem * (1.0 - sem))
        Ls = g_sem @ self.Wp
        eK = st['eK']
        if self.trace_fp16 and self.grad_einsum16:
            # [O8a] fp16 einsum 归约（内部 fp16 累加）：实测 79µs vs mul-sum 333µs（4.2×）
            Ls16 = Ls.half()
            gWg = torch.einsum('bh,bhd->hd', Ls16, eK[0]).float()
            gWd = torch.einsum('bh,bhd->hd', Ls16, eK[1]).float()
            gB = torch.einsum('bh,bhd->hd', Ls16, eK[2]).float()
        elif self.trace_fp16:
            # 旧路径：fp16 迹 -> 逐元素乘 + fp32 累加归约（更稳但更慢）
            Ls16 = Ls.half()
            gWg = (Ls16.unsqueeze(-1) * eK[0]).sum(0, dtype=torch.float32)
            gWd = (Ls16.unsqueeze(-1) * eK[1]).sum(0, dtype=torch.float32)
            gB = (Ls16.unsqueeze(-1) * eK[2]).sum(0, dtype=torch.float32)
        else:
            gWg = torch.einsum('bh,bhd->hd', Ls, eK[0])
            gWd = torch.einsum('bh,bhd->hd', Ls, eK[1])
            gB = torch.einsum('bh,bhd->hd', Ls, eK[2])
        grads = [
            ('W_g', gWg),
            ('b_g', torch.einsum('bh,bh->h', Ls, st['e_bg'])),
            ('W_d', gWd),
            ('b_d', torch.einsum('bh,bh->h', Ls, st['e_bd'])),
            ('B', gB),
            ('Wp', g_sem.t() @ s),
            ('bp', g_sem.sum(0)),
            ('head', e.t() @ sem),
            ('bout', e.sum(0)),
        ]
        self._apply(grads)

    def _apply(self, grads):
        if not torch.cuda.is_current_stream_capturing():
            self._set_bias_factors()   # [O9] 图捕捉时跳过（由回放器在图外设置）
        b1c_inv = self._b1c_inv
        b2c = self._b2c
        names = [n for n, _ in grads]
        gs = [g for _, g in grads]
        params = [self.params()[n] for n in names]
        mm = [self._meta[n][0] for n in names]
        vv = [self._meta[n][1] for n in names]
        if self.use_foreach:
            # [O2] 批量融合：9 参数 × ~7 算子 -> ~7 次 foreach 调用
            torch._foreach_mul_(mm, self.beta1)
            torch._foreach_add_(mm, gs, alpha=1.0 - self.beta1)
            torch._foreach_mul_(vv, self.beta2)
            torch._foreach_addcmul_(vv, gs, gs, value=1.0 - self.beta2)
            mhat = torch._foreach_mul(mm, b1c_inv)
            vhat = torch._foreach_sqrt(torch._foreach_div(vv, b2c))
            torch._foreach_add_(vhat, self.eps)
            upd = torch._foreach_div(mhat, vhat)
            if self.wd > 0.0:
                p2 = [p for p in params if p.dim() == 2]
                if p2:
                    torch._foreach_mul_(p2, 1.0 - self.wd)
            if self.adapt:
                # [O11] 信赖域步长：位移 ≤ s·ρ·R（R=全体模型参数含冻结嵌入 E 的 RMS，自派生；ρ 无量纲）——
                # 均匀绝对步长：保留旧绝对-η 域的逐张量有效相对步长结构（逐张量相对尺度会放大
                # 大尺度张量如 b_d 的相对位移近 20×，实测导致门控崩溃，已弃用）。
                # R 含 E：标定口径 η_cal/R_all = 2e-3/0.41 ≈ ρ；仅算可训练参数会被 b_d 的 rms≈3.3
                # 独占失真（基准 η 偏小 ~3×，实测 valCE 劣化 0.1+），已修正。
                # [O11-修] 步长基准 = 建立时冻结的 R（self._r0）；逐步实测 R 会随权重漂移正反馈放大
                coef = self._s * TRUST_RHO * self._r0
                steps = []
                for k, p in enumerate(params):
                    st_k = upd[k].clamp(-DIR_CLAMP, DIR_CLAMP) * coef
                    steps.append(st_k)
                    p.add_(st_k, alpha=-1.0)
            else:
                torch._foreach_add_(params, upd, alpha=-self.eta)
        else:
            if self.adapt:
                coef = self._s * TRUST_RHO * self._r0
            for name, grad in grads:
                param = self.params()[name]
                m, v = self._meta[name]
                m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
                mhat = m * b1c_inv
                vhat = v / b2c
                upd = mhat / (vhat.sqrt() + self.eps)
                if self.wd > 0.0 and param.dim() == 2:
                    param.mul_(1.0 - self.wd)
                if self.adapt:
                    # [O11] 逐参数信赖域（非 foreach 慢路径，同数学）
                    st_k = upd.clamp(-DIR_CLAMP, DIR_CLAMP) * coef
                    param.add_(st_k, alpha=-1.0)
                else:
                    param.add_(upd, alpha=-self.eta)
        if self._ema is not None:
            t = self.steps + 1
            b = min(EMA_DECAY, (1.0 + t) / (10.0 + t))   # [O15] 暖启动：早期近似滑动平均，避免滞后
            es = list(self._ema.values())
            torch._foreach_mul_(es, b)
            torch._foreach_add_(es, list(self.params().values()), alpha=1.0 - b)
        self.steps += 1

    # ---- 训练步：返回 (logits, 掩码NLL和, 计数)；[O4] NLL 复用同一次前向 ----
    @torch.no_grad()
    def step(self, tok, tgt, st, train=True, mask=None):
        logits, sem = self._cell(tok, st, trace=train)
        if not train:
            return logits, None, None
        nll = torch.logsumexp(logits, dim=-1) - logits.gather(1, tgt.view(-1, 1)).squeeze(1)
        p = torch.softmax(logits, dim=-1)
        B = tok.numel()
        e = p
        if BIO_LSM > 0.0:
            e = e - BIO_LSM / self.vocab
        e[self._arange(B), tgt] = e[self._arange(B), tgt] - (1.0 - BIO_LSM)
        if mask is None:
            e = e / B
            nll_sum = nll.sum()
            cnt = torch.tensor(float(B), device=DEV)
        else:
            mm = mask.to(e.dtype).unsqueeze(1)
            e = e * mm / mm.sum().clamp(min=1.0)
            nll_sum = (nll * mask).sum()
            cnt = mask.sum()
        self._learn(sem, st['s'], st, e)
        if self.adapt:
            # [O11] 反馈统计：窗口累计为张量运算（图回放可推进）；tick 由 eager/回放器在捕捉外触发
            l_step = nll_sum / cnt.clamp(min=1.0)
            self._win_sum.add_(l_step)
            self._win_sq.add_(l_step * l_step)
            self._win_cnt += 1
            if not torch.cuda.is_current_stream_capturing() and self._win_cnt >= self._win_len:
                self._controller_tick()
        return logits, nll_sum, cnt

    # ---- 评估：不维护资格迹 ----
    @torch.no_grad()
    def eval_masked(self, streams, targets, mask):
        streams = streams.to(DEV)
        targets = targets.to(DEV)
        mask = mask.to(DEV)
        ns, T = streams.shape
        st = self.init_stream_state(ns)
        tot = torch.zeros((), device=DEV)
        cnt = torch.zeros((), device=DEV)
        for t in range(T):
            logits, _, _ = self.step(streams[:, t], targets[:, t], st, train=False)
            nll = torch.logsumexp(logits, dim=-1) - logits.gather(1, targets[:, t].view(-1, 1)).squeeze(1)
            tot = tot + (nll * mask[:, t]).sum()
            cnt = cnt + mask[:, t].sum()
        return float(tot / cnt.clamp(min=1.0))

    # ---- 生成（流式回调；不维护资格迹） ----
    @torch.no_grad()
    def generate(self, tokenizer, prompt_ids, n_new=256, temp=None, top_p=None,
                 seed=0, on_token=None):
        temp = TEMPERATURE if temp is None else temp
        top_p = TOP_P if top_p is None else top_p
        g = torch.Generator().manual_seed(seed)
        st = self.init_stream_state(1)
        logits = None
        for t in prompt_ids:
            logits, _ = self._cell(torch.tensor([int(t)], device=DEV), st, trace=False)
        out = []
        for _ in range(n_new):
            z = logits[0].clone()
            z[tokenizer.ids['<|pad|>']] = -1e9
            z[tokenizer.ids['<|unk|>']] = -1e9
            if temp <= 0.01:
                nxt = int(z.argmax())
            else:
                p = torch.softmax(z / temp, dim=-1)
                sp, si = torch.sort(p, descending=True)
                cs = torch.cumsum(sp, dim=0)
                cut = cs > top_p
                cut[0] = False
                sp[cut] = 0.0
                ssum = sp.sum()
                sp = sp / (ssum if ssum > 0 else 1.0)
                nxt = int(si[torch.multinomial(sp.cpu(), 1, generator=g)])
            if nxt in (tokenizer.ids['<|end|>'], tokenizer.ids['<|eos|>']):
                break
            out.append(nxt)
            if on_token is not None:
                on_token(nxt)
            logits, _ = self._cell(torch.tensor([nxt], device=DEV), st, trace=False)
        return out

    # ---- 完成段逐 token 对数概率（供 RL/GRPO 旧策略、评估用；不维护资格迹） ----
    @torch.no_grad()
    def completion_logps(self, prompt_ids, completion_ids):
        if not completion_ids:
            return torch.empty(0, device=DEV)
        st = self.init_stream_state(1)
        for t in prompt_ids:
            self._cell(torch.tensor([int(t)], device=DEV), st, trace=False)
        lps = []
        for a in completion_ids:
            logits, _ = self._cell(torch.tensor([int(a)], device=DEV), st, trace=False)
            lp = torch.log_softmax(logits, dim=-1)[0, int(a)]
            lps.append(lp)
        return torch.stack(lps)

    # ---- 局部规则策略梯度更新（RL/GRPO 共用；等价"加权 CE"，零 autograd）
    #      e_t = w_t · (p_t - onehot(a_t)) / n；w_t 由调用方按原版公式给出（优势/ratio/anchor）
    @torch.no_grad()
    def pg_update(self, prompt_ids, completion_ids, weight_fn, eta=None):
        """[O13] 优势按滚动 RMS 归一化（w/√rms²）+ 复用自适应步长：
        更新量不随奖励形状漂移——ULTRA_*_LEARNING_RATE 无需按奖励形状手调（自适应开启时忽略）。"""
        n = max(1, len(completion_ids))
        st = self.init_stream_state(1)
        for t in prompt_ids:
            self._cell(torch.tensor([int(t)], device=DEV), st, trace=True)
        old_eta = self.eta
        if eta is not None and not self.adapt:
            self.eta = float(eta)
        try:
            for j, a in enumerate(completion_ids):
                tok = torch.tensor([int(a)], device=DEV)
                logits, sem = self._cell(tok, st, trace=True)
                p = torch.softmax(logits[0], dim=-1)
                w_raw = float(weight_fn(j, p))
                w = w_raw / max(float(self._w_rms2.clamp(min=1e-12).sqrt()), 1e-12)
                alpha = 1.0 / (self._win_len + 1.0)   # 滚动统计窗与反馈窗同源（自派生）
                self._w_rms2.mul_(1.0 - alpha).add_(w_raw * w_raw, alpha=alpha)
                e = p
                e[int(a)] -= 1.0
                e.mul_(w / n)
                self._learn(sem, st['s'], st, e.unsqueeze(0))
        finally:
            self.eta = old_eta


# ============================================================
# 检查点（brain.pt 口径与原版一致：{'model_state_dict': ..., 'epoch': ...}）
# ============================================================
def model_state_dict(model):
    d = {k: v.detach().cpu().clone() for k, v in model.params().items()}
    d['E'] = model.E.detach().cpu().clone()
    return d


def get_next_checkpoint_id() -> str:
    timestamp = int(time.time() * 1000)
    rand = random.randint(0, 10000)
    return '%d_%d' % (timestamp, rand)


def _ema_swap_in(model):
    """[O15] 临时把 EMA 权重换入推理/保存路径，返回可还原的当前权重快照。"""
    if getattr(model, '_ema', None) is None:
        return None
    cur = {k: p.detach().clone() for k, p in model.params().items()}
    for k, p in model.params().items():
        p.copy_(model._ema[k])
    return cur


def _ema_swap_out(model, cur):
    if cur is None:
        return
    for k, p in model.params().items():
        p.copy_(cur[k])


def _ema_finalize(model):
    """训练结束时把平均权重固化进模型，使会话内模型与保存/部署一致。"""
    if getattr(model, '_ema', None) is None:
        return
    for k, p in model.params().items():
        p.copy_(model._ema[k])


def save_checkpoint(model, epoch, custom_filename=None):
    if custom_filename is None:
        filename = 'checkpoint_%s.pt' % get_next_checkpoint_id()
    else:
        filename = custom_filename
    cur = _ema_swap_in(model)
    save_dict = {
        'model_state_dict': model_state_dict(model),
        'epoch': epoch,
        'bio_meta': {'h': model.h, 'd': model.d, 'm': model.m,
                     'vocab': model.vocab, 'steps': model.steps},
        'ema_decay': EMA_DECAY,
        'lsm': BIO_LSM,
    }
    torch.save(save_dict, filename)
    _ema_swap_out(model, cur)
    log_and_print('保存 checkpoint 至 %s (epoch %d)' % (filename, epoch))
    return filename


def _load_state_into(model, checkpoint):
    state = checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
        meta = checkpoint.get('bio_meta')
        if meta and (meta.get('h') != model.h or meta.get('d') != model.d
                     or meta.get('m') != model.m):
            raise ValueError('检查点形状与当前内核超参不符: %s vs h=%d d=%d m=%d'
                             % (meta, model.h, model.d, model.m))
    for k, p in model.params().items():
        if k in state:
            p.copy_(state[k].to(DEV))
    if 'E' in state:
        model.E.copy_(state['E'].to(DEV))
    if getattr(model, 'adapt', False) and hasattr(model, '_compute_r0'):
        model._r0 = model._compute_r0()   # [O11-修] 载入后刷新步长基准
    if getattr(model, '_ema', None) is not None:
        for k, p in model.params().items():
            model._ema[k].copy_(p)


def load_brain_pt_into(model) -> bool:
    """将 brain.pt 加载到模型；checkpoint 临时放内存，写入后立即释放。返回 False 表示 brain.pt 不存在。"""
    if not os.path.exists('brain.pt'):
        return False
    checkpoint = torch.load('brain.pt', map_location='cpu', weights_only=False)
    _load_state_into(model, checkpoint)
    del checkpoint
    if DEV.type == 'cuda':
        torch.cuda.empty_cache()
    return True


# ============================================================
# 文件选择（原样）
# ============================================================
def select_training_file():
    """扫描当前目录下 .txt 和 .jsonl 文件，让用户选择数据集文件。"""
    files = []
    for f in sorted(os.listdir('.')):
        if f.startswith('.'):
            continue
        if f.endswith('.txt') or f.endswith('.jsonl'):
            try:
                size = os.path.getsize(f)
            except OSError:
                size = 0
            files.append((f, size))
    if not files:
        log_and_print('当前目录下未找到 .txt 或 .jsonl 文件。')
        return None, None
    print('\n可用数据集文件：')
    for i, (name, size) in enumerate(files, 1):
        if size >= 1024 * 1024:
            size_str = '%.1f MB' % (size / 1024 / 1024)
        elif size >= 1024:
            size_str = '%.1f KB' % (size / 1024)
        else:
            size_str = '%d B' % size
        print('  %d. %s (%s)' % (i, name, size_str))
    while True:
        raw = input('请输入文件序号: ').strip()
        try:
            idx = int(raw)
        except ValueError:
            print('输入无效，请输入数字。')
            continue
        if idx < 1 or idx > len(files):
            print('序号超出范围，请输入 1-%d。' % len(files))
            continue
        selected, _ = files[idx - 1]
        ext = os.path.splitext(selected)[1].lower()
        log_and_print('已选择文件: %s' % selected)
        return selected, ext


# ============================================================
# LINUX 实验性加速：内核单步编译（失败/异常一律静默回退 eager，数学不变）
# ============================================================
class _MaybeCompiledStep:
    def __init__(self, fn):
        self.fn = fn
        self.c = None
        self.failed = False

    def __call__(self, *a, **kw):
        if self.failed:
            return self.fn(*a, **kw)
        try:
            if self.c is None:
                self.c = torch.compile(self.fn, dynamic=False,
                                       mode='max-autotune-no-cudagraphs')
            return self.c(*a, **kw)
        except Exception as e:
            log_and_print('[LINUX] compiled step 失败，回退 eager：%s' % e)
            self.failed = True
            return self.fn(*a, **kw)


def linux_maybe_compile_step(model):
    if LINUX_COMPILE_STEP:
        model.step = _MaybeCompiledStep(model.step)
        log_and_print('[LINUX] 已挂载编译包装（首步触发编译，失败自动回退）')


# ============================================================
# 训练（外壳流程对齐原版：轮次/验证/最佳保存/早停/Ctrl+C 落盘）
# ============================================================
def _pick_streams():
    return STREAMS_CUDA if DEV.type == 'cuda' else STREAMS_CPU


def _eval_streams(model, ids, mask, ns=EVAL_STREAMS):
    ns = min(ns, max(1, len(ids) // 8))   # [O10] 数据太短时自动降流数
    st, tg, mk = to_streams(ids, mask, ns)
    cur = _ema_swap_in(model)
    try:
        return model.eval_masked(st, tg, mk)
    finally:
        _ema_swap_out(model, cur)


# ============================================================
# [O9] 训练步 CUDA Graph 回放器（与 eager 同进程逐位等价；捕捉失败自动回退）
# ============================================================
class _GraphStepRunner:
    """把一次训练步（含局部更新）抓成 CUDA Graph；回放免 Python 派发与启动开销。

    用法：
        runner = _GraphStepRunner(model, n_streams)   # 失败时 runner.ok=False -> 回退 eager
        runner.run(tok_col, tgt_col, mask_col)        # 每步一次，返回 (logits, nll_sum, cnt)
    说明：
      - 捕捉前自动快照并还原训练状态（预热/抓取阶段的参数前进不回写模型）；
        流状态由 runner 持有并随回放推进，reset_state() 可清零重开（对齐"每轮全新状态"）。
      - run() 内部推进 model.steps（偏差校正常量由图外 _set_bias_factors 写入）。
      - 限制：step 内不得有 CPU 同步（本内核已无 .item()）。
    """

    def __init__(self, model, ns, warmup=3):
        self.model = model
        self.ns = ns
        self.ok = False
        self.err = ''
        self.st = None
        self.out = None
        self.g = None
        try:
            if DEV.type != 'cuda':
                self.err = '非 CUDA 设备'
                return
            self.tok = torch.zeros(ns, dtype=torch.long, device=DEV)
            self.tgt = torch.zeros(ns, dtype=torch.long, device=DEV)
            self.mask = torch.ones(ns, device=DEV)
            snap = model._snapshot_train_state()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                st0 = model.init_stream_state(ns)
                for _ in range(warmup):
                    model._set_bias_factors()
                    model.step(self.tok, self.tgt, st0, train=True, mask=self.mask)
                torch.cuda.synchronize()
            model._restore_train_state(snap)
            self.st = model.init_stream_state(ns)
            self.g = torch.cuda.CUDAGraph()
            # 注意：偏差常量必须在图外设置（放入图内会被回放覆盖为捕捉时的值）
            model._set_bias_factors()
            with torch.cuda.graph(self.g):
                self.out = model.step(self.tok, self.tgt, self.st, train=True,
                                      mask=self.mask)
            model._restore_train_state(snap)
            for k in self.st:
                self.st[k].zero_()
            model._set_bias_factors()
            self.ok = True
        except Exception as e:
            self.ok = False
            self.err = str(e)[:160]

    def reset_state(self):
        if self.st is not None:
            for k in self.st:
                self.st[k].zero_()

    @torch.no_grad()
    def run(self, tok_col, tgt_col, mask_col):
        self.tok.copy_(tok_col)
        self.tgt.copy_(tgt_col)
        if mask_col is None:
            self.mask.fill_(1.0)
        else:
            self.mask.copy_(mask_col)
        self.model._set_bias_factors()
        self.g.replay()
        self.model.steps += 1
        if self.model.adapt:
            # [O11] 窗口计数与 tick 是 Python 侧（回放不重放 Python）；窗口累计是张量、已被回放推进
            self.model._win_cnt += 1
            if self.model._win_cnt >= self.model._win_len:
                self.model._controller_tick()
        return self.out


def train_from_scratch(model, tokenizer, data_path, batch_size, epochs, lr,
                       weight_decay, grad_clip, log_interval,
                       accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                       file_type='jsonl'):
    model.eta = float(lr)
    model.wd = float(weight_decay)
    log_and_print('BioELM 局部规则内核：开始流式训练。数据集: %s' % data_path)
    log_and_print('（接口说明：batch_size/grad_clip/accumulation 为原版壳保留参数，在线逐列内核不使用；'
                  'AMP/层监控随原内核移除；CUDA Graph 见 GRAPH_STEP）')
    if model.adapt:
        log_and_print('[O11] 自适应学习率已启用：无需按模型规模手调（LEARNING_RATE 仅作接口/回退）；'
                      '信赖域 ρ=%.0e，反馈窗由电路记忆半衰期自派生' % TRUST_RHO)

    try:
        import psutil
        process = psutil.Process(os.getpid())
        _has_psutil = True
    except ImportError:
        process = None
        _has_psutil = False

    log_and_print('--- 从头训练开始 (%s) ---' % data_path)

    if file_type == '.txt':
        ids = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.extend(tokenizer.encode(normalize_text(line) + '\n'))
        tr_ids, tr_mask = ids, [1] * len(ids)
        va_ids = va_mask = None
        log_and_print('txt 流式模式：%d tokens，不验证不早停。' % len(tr_ids))
    else:
        tr_ids, tr_mask, va_ids, va_mask = build_corpus(tokenizer, data_path)

    if not tr_ids:
        log_and_print('训练语料为空，终止。')
        return

    linux_maybe_compile_step(model)
    ns = _pick_streams()
    if len(tr_ids) < ns * 8:
        ns = max(1, len(tr_ids) // 8)
    total_learnable = sum(tr_mask)

    best_val_ppl = float('inf')
    patience_counter = 0
    steps_total = 0
    t0_all = time.time()

    runner = None
    if GRAPH_STEP and DEV.type == 'cuda':
        runner = _GraphStepRunner(model, ns)
        if runner.ok:
            log_and_print('[O9] CUDA Graph 训练步已启用（整步回放，免 Python 派发）')
        else:
            log_and_print('[O9] CUDA Graph 不可用，回退 eager：%s' % runner.err)
            runner = None

    interrupted = False
    try:
        for epoch in range(epochs):
            rotate = 0 if epoch == 0 else random.randrange(0, max(1, len(tr_ids)))
            st, tg, mk = to_streams(tr_ids, tr_mask, ns, rotate=rotate)
            st_t = st.t().contiguous().to(DEV)
            tg_t = tg.t().contiguous().to(DEV)
            mk_t = mk.t().contiguous().to(DEV)
            T = st_t.shape[0]
            if runner is not None:
                runner.reset_state()   # 对齐 eager：每轮以全新流状态开始
                state = None
            else:
                state = model.init_stream_state(ns)
            run_loss = torch.zeros((), device=DEV)
            run_cnt = torch.zeros((), device=DEV)
            learned_total = torch.zeros((), device=DEV)
            epoch_steps = 0
            log_and_print('轮次 %d/%d 开始：%d 条流 × %d 步（%s 上在线推进）'
                          % (epoch + 1, epochs, ns, T, str(DEV)))
            for t in range(T):
                if runner is not None:
                    _, nll, cnt = runner.run(st_t[t], tg_t[t], mk_t[t])
                else:
                    _, nll, cnt = model.step(st_t[t], tg_t[t], state, train=True, mask=mk_t[t])
                run_loss = run_loss + nll
                run_cnt = run_cnt + cnt
                learned_total = learned_total + cnt
                steps_total += 1
                epoch_steps += 1
                if log_interval > 0 and epoch_steps % log_interval == 0:
                    loss_v = float(run_loss / run_cnt.clamp(min=1.0))
                    ppl_v = math.exp(loss_v) if loss_v < 50 else float('inf')
                    mem_str = ''
                    if _has_psutil and process is not None:
                        try:
                            mem_str = ' Memory: %.1f MB' % (process.memory_info().rss / 1024 ** 2)
                        except Exception:
                            mem_str = ''
                    learned_v = float(learned_total)
                    pct_str = ' (%.1f%%)' % (learned_v * 100.0 / max(1, total_learnable))
                    log_and_print('Epoch %d/%d Step %d/%d Loss: %.4f PPL: %.2f '
                                  'ETA: %.2e Learned: %d%s%s'
                                  % (epoch + 1, epochs, epoch_steps, T, loss_v, ppl_v,
                                     model.eta, int(learned_v), pct_str, mem_str))
                    run_loss = torch.zeros((), device=DEV)
                    run_cnt = torch.zeros((), device=DEV)
                if EMPTY_CACHE_INTERVAL > 0 and steps_total % EMPTY_CACHE_INTERVAL == 0 \
                        and DEV.type == 'cuda':
                    torch.cuda.empty_cache()

            if va_ids:
                ce_v = _eval_streams(model, va_ids, va_mask)
                ppl_v = math.exp(ce_v) if ce_v < 50 else float('inf')
                log_and_print('Epoch %d 验证 Loss: %.4f PPL: %.2f' % (epoch + 1, ce_v, ppl_v))
                with open(LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write('Epoch %d | Val Loss: %.4f | Val PPL: %.2f\n' % (epoch + 1, ce_v, ppl_v))
                if ppl_v < best_val_ppl:
                    best_val_ppl = ppl_v
                    patience_counter = 0
                    best_filename = 'best_epoch%d_ppl%.2f.pt' % (epoch + 1, ppl_v)
                    save_checkpoint(model, epoch + 1, custom_filename=best_filename)
                    log_and_print('保存最佳模型至 %s' % best_filename)
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOP_PATIENCE:
                        log_and_print('早停触发，停止训练')
                        break
            else:
                log_and_print('Epoch %d 无验证集，跳过验证与早停。' % (epoch + 1))
            save_checkpoint(model, epoch + 1,
                            custom_filename='checkpoint_epoch%d.pt' % (epoch + 1))
    except KeyboardInterrupt:
        interrupted = True
        log_and_print('\n训练被用户中断 (Ctrl+C)，正在保存当前模型...')
        save_checkpoint(model, 0, custom_filename='interrupted_model.pt')
        log_and_print('模型已保存至 interrupted_model.pt，返回主菜单。')
        return

    if not interrupted:
        _ema_finalize(model)
        log_and_print('训练完成（%d 步，%.1fs）'
                      % (steps_total, time.time() - t0_all))


# ============================================================
# 对话循环（原样：jsonl / txt 两种模式，历史窗口管理）
# ============================================================
def chat_loop(model, tokenizer, mode='jsonl'):
    MAX_HISTORY_TOKENS = MAX_SEQ_LEN

    if mode == 'txt':
        log_and_print('进入纯文本续写模式（txt），输入 exit 退出，reset 重置上下文。')
        context_ids = []
        while True:
            user_input = input('\n输入: ')
            if user_input.lower() == 'exit':
                break
            if user_input.lower() == 'reset':
                context_ids = []
                log_and_print('上下文已重置。')
                continue
            new_ids = tokenizer.encode(user_input)
            context_ids.extend(new_ids)
            if len(context_ids) > MAX_HISTORY_TOKENS:
                context_ids = context_ids[-MAX_HISTORY_TOKENS:]
            generated_ids = []

            def stream_token(token_id):
                generated_ids.append(token_id)
                text = tokenizer.decode([token_id])
                if text:
                    print(text, end='', flush=True)

            model.generate(tokenizer, context_ids, n_new=MAX_SEQ_LEN,
                           temp=TEMPERATURE, top_p=TOP_P, on_token=stream_token)
            context_ids.extend(generated_ids)
            if len(context_ids) > MAX_HISTORY_TOKENS:
                context_ids = context_ids[-MAX_HISTORY_TOKENS:]
            print()
        print()
    else:
        log_and_print('进入对话模式（jsonl），输入 exit 退出，reset 重置上下文。')
        history_turns = []
        while True:
            user_input = input('\n用户: ')
            if user_input.lower() == 'exit':
                break
            if user_input.lower() == 'reset':
                history_turns = []
                log_and_print('上下文已重置。')
                continue
            user_ids = [tokenizer.user_token_id] + tokenizer.encode(user_input) + [tokenizer.end_token_id]
            max_input_len = 10000000 - 200  # 无上限
            max_history_len = max_input_len - (1 + len(user_ids) + 1)
            history_len = sum(len(t) for t in history_turns)
            while history_len > max_history_len and len(history_turns) > 1:
                history_len -= len(history_turns.pop(0))
            flat_history = [tok for turn in history_turns for tok in turn]
            full_input = [tokenizer.bos_token_id] + flat_history + user_ids + [tokenizer.assistant_token_id]
            generated_ids = []
            print('助手: ', end='', flush=True)
            _sdec = StreamDecoder(tokenizer)

            def stream_token_jsonl(token_id):
                generated_ids.append(token_id)
                text = _sdec.push(token_id)
                if text:
                    print(text, end='', flush=True)

            model.generate(tokenizer, full_input, n_new=MAX_SEQ_LEN,
                           temp=TEMPERATURE, top_p=TOP_P, on_token=stream_token_jsonl)
            _tail = _sdec.flush()
            if _tail:
                print(_tail, end='', flush=True)
            print()
            history_turns.append(user_ids)
            history_turns.append([tokenizer.assistant_token_id] + generated_ids)
            total = sum(len(t) for t in history_turns)
            while total > MAX_HISTORY_TOKENS and len(history_turns) > 1:
                total -= len(history_turns.pop(0))


# ============================================================
# Ultra：规则验证拒绝采样 / 规则奖励 RL / GRPO（菜单流程原样；更新层为局部规则）
# ============================================================
def ultra_read_int(prompt: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = input('%s [%d]: ' % (prompt, default)).strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print('请输入不小于 %d 的整数。' % minimum)


def ultra_read_float(prompt: str, default: float, minimum: float = 0.0) -> float:
    while True:
        raw = input('%s [%s]: ' % (prompt, default)).strip()
        if not raw:
            return default
        try:
            value = float(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print('请输入不小于 %s 的数字。' % minimum)


def ultra_select_problem_file():
    files = []
    for name in sorted(os.listdir('.')):
        if name.startswith('.') or not name.endswith('.jsonl'):
            continue
        try:
            size = os.path.getsize(name)
        except OSError:
            size = 0
        files.append((name, size))
    if not files:
        log_and_print('当前目录没有可用的 .jsonl 题库。')
        return None
    print('\n可用 JSONL 题库：')
    for i, (name, size) in enumerate(files, 1):
        print('  %d. %s (%.1f KB)' % (i, name, size / 1024))
    while True:
        raw = input('请输入题库序号: ').strip()
        try:
            index = int(raw)
            if 1 <= index <= len(files):
                return files[index - 1][0]
        except ValueError:
            pass
        print('请输入 1-%d。' % len(files))


def ultra_iter_problems(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                log_and_print('跳过题库第 %d 行：JSON 解析失败: %s' % (line_no, e))
                continue
            try:
                msgs = _messages_from_record(item)
            except Exception as e:
                log_and_print('跳过题库第 %d 行：%s' % (line_no, e))
                continue
            question = '\n'.join(str(m.get('content', '')) for m in msgs
                                 if m.get('role') in ('user', 'system'))
            answers = item.get('answers', item.get('answer'))
            if not question.strip() or answers is None:
                log_and_print('跳过题库第 %d 行：v5 题库需要 messages 与 answer/answers。' % line_no)
                continue
            if not isinstance(answers, list):
                answers = [answers]
            answers = [str(answer) for answer in answers if str(answer).strip()]
            if not answers:
                log_and_print('跳过题库第 %d 行：没有可用答案。' % line_no)
                continue
            yield {
                'question': question,
                'messages': msgs or None,
                'answers': answers,
                'type': str(item.get('type', 'auto')).lower(),
                'tools': item.get('tools'),
                'line_no': line_no,
            }


def ultra_count_problems(path: str) -> int:
    n = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def ultra_iter_problems_shuffled(path: str, buffer_size: int = 10000, seed: int = 0):
    rng = random.Random(seed)
    buf = []
    for p in ultra_iter_problems(path):
        buf.append(p)
        if len(buf) >= buffer_size:
            rng.shuffle(buf)
            while buf:
                yield buf.pop()
    if buf:
        rng.shuffle(buf)
        while buf:
            yield buf.pop()


def ultra_infer_answer_type(problem) -> str:
    answer_type = problem.get('type', 'auto')
    if answer_type != 'auto':
        return answer_type
    answers = [answer.strip() for answer in problem['answers']]
    if answers and all(re.fullmatch(r'[A-Za-z]', answer) for answer in answers):
        return 'choice'
    if answers and all(re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?', answer.replace(',', ''))
                       for answer in answers):
        return 'number'
    return 'text'


def ultra_normalize_answer(value: str, answer_type: str) -> str:
    text = unicodedata.normalize('NFKC', str(value)).strip()
    if answer_type == 'choice':
        choices = re.findall(r'(?<![A-Za-z])[A-Za-z](?![A-Za-z])', text)
        return choices[-1].upper() if choices else text.upper()
    if answer_type == 'number':
        matches = re.findall(r'[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?', text.replace(',', ''))
        if matches:
            raw = matches[-1]
            try:
                if '/' in raw:
                    left, right = raw.split('/', 1)
                    value_fraction = Fraction(Decimal(left)) / Fraction(Decimal(right))
                else:
                    value_fraction = Fraction(Decimal(raw))
                return '%d/%d' % (value_fraction.numerator, value_fraction.denominator)
            except (InvalidOperation, ZeroDivisionError):
                pass
    text = re.sub(r'\s+', '', text.lower())
    return text.strip('。.!！?？,，;；:：')


def ultra_extract_final_answer(response: str) -> str:
    boxed = re.findall(r'\\boxed\s*\{([^{}]+)\}', response)
    if boxed:
        return boxed[-1].strip()
    marked = re.findall(r'(?:最终答案|答案|final\s+answer)\s*[:：]\s*([^\r\n]+)', response,
                        flags=re.IGNORECASE)
    if marked:
        return marked[-1].strip()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ''


def ultra_rule_reward(response: str, problem):
    _reasoning, body = _split_reasoning(response)      # 只看正文，思考内容不参与判分
    predicted = ultra_extract_final_answer(body)
    answer_type = ultra_infer_answer_type(problem)
    predicted_normalized = ultra_normalize_answer(predicted, answer_type)
    accepted = {ultra_normalize_answer(answer, answer_type) for answer in problem['answers']}
    reward = 1.0 if predicted_normalized in accepted else 0.0
    return reward, predicted


def ultra_make_prompt(problem) -> str:
    return '%s%s' % (problem['question'], RL_PROMPT_SUFFIX)


def ultra_prompt_ids(tokenizer, prompt: str):
    msgs = []
    if RL_SYSTEM_PROMPT.strip():
        msgs.append({'role': 'system', 'content': RL_SYSTEM_PROMPT})
    msgs.append({'role': 'user', 'content': prompt})
    text, _spans = render_messages(msgs, add_generation_prompt=True)
    ids, _offs = _encode_with_offsets(tokenizer, text)
    return ids


def ultra_generate_one(model, tokenizer, prompt_ids, max_new_tokens: int):
    generated_ids = model.generate(tokenizer, prompt_ids, n_new=max_new_tokens,
                                   temp=TEMPERATURE, top_p=TOP_P)
    return generated_ids, tokenizer.decode(generated_ids)


def ultra_write_sft_line(f, prompt: str, response: str, meta: dict = None, reasoning_content: str = ''):
    """写一条 SFT 样本（v5：messages + meta，可被本文件菜单 1 直接读回）。"""
    assistant = {'role': 'assistant', 'content': response}
    if reasoning_content:
        assistant['reasoning_content'] = reasoning_content
    record = {'messages': [{'role': 'user', 'content': prompt}, assistant]}
    if meta:
        record['meta'] = dict(meta)
    f.write(json.dumps(record, ensure_ascii=False) + '\n')
    f.flush()


def ultra_rejection_sampling(model, tokenizer, problem_path: str,
                             output_path: str, samples_per_question: int,
                             max_new_tokens: int):
    if os.path.abspath(problem_path) == os.path.abspath(output_path):
        raise ValueError('输出文件不能覆盖输入题库。')
    total_problems = ultra_count_problems(problem_path)
    if total_problems == 0:
        log_and_print('题库没有可用题目。')
        return
    kept = 0
    tried = 0
    seen = set()
    with open(output_path, 'w', encoding='utf-8') as out:
        for index, problem in enumerate(ultra_iter_problems(problem_path), 1):
            prompt = ultra_make_prompt(problem)
            prompt_ids = ultra_prompt_ids(tokenizer, prompt)
            local_kept = 0
            for _ in range(samples_per_question):
                generated_ids, response = ultra_generate_one(model, tokenizer, prompt_ids, max_new_tokens)
                tried += 1
                reward, predicted = ultra_rule_reward(response, problem)
                key = (problem['line_no'], response)
                if reward > 0 and key not in seen:
                    ultra_write_sft_line(out, prompt, response, {
                        'reward': float(reward), 'predicted': predicted, 'from': 'rs',
                        'line_no': problem['line_no']})
                    seen.add(key)
                    kept += 1
                    local_kept += 1
            log_and_print('[RS %d/%d] 正确保留 %d/%d' % (index, total_problems, local_kept, samples_per_question))
    log_and_print('拒绝采样完成：尝试 %d 个回答，保留 %d 个正确回答，输出至 %s' % (tried, kept, output_path))


def ultra_completion_log_probs(model, prompt_ids, completion_ids):
    """完成段逐 token log 概率（前向，无更新）——GRPO 旧策略基准。"""
    return model.completion_logps(prompt_ids, completion_ids)


def ultra_save_model_only(model, filename: str, epoch: int):
    torch.save({'model_state_dict': model_state_dict(model), 'epoch': epoch,
                'bio_meta': {'h': model.h, 'd': model.d, 'm': model.m,
                             'vocab': model.vocab, 'steps': model.steps}}, filename)
    log_and_print('仅保存模型至 %s' % filename)


def ultra_grpo_train(model, tokenizer, problem_path: str,
                     positive_output_path: str, checkpoint_prefix: str,
                     epochs: int, group_size: int, max_new_tokens: int,
                     lr: float, inner_steps: int, clip_epsilon: float,
                     old_policy_penalty: float):
    if os.path.abspath(problem_path) == os.path.abspath(positive_output_path):
        raise ValueError('GRPO 正确样本输出文件不能覆盖输入题库。')
    total_problems = ultra_count_problems(problem_path)
    if total_problems == 0:
        log_and_print('题库没有可用题目。')
        return
    seen_positive = set()
    total_updates = 0
    with open(positive_output_path, 'w', encoding='utf-8') as positive_out:
        for epoch in range(epochs):
            for index, problem in enumerate(ultra_iter_problems_shuffled(problem_path, buffer_size=10000, seed=epoch), 1):
                prompt = ultra_make_prompt(problem)
                prompt_ids = ultra_prompt_ids(tokenizer, prompt)
                completions = []
                responses = []
                rewards = []
                for _ in range(group_size):
                    completion_ids, response = ultra_generate_one(model, tokenizer, prompt_ids, max_new_tokens)
                    reward, predicted = ultra_rule_reward(response, problem)
                    completions.append(completion_ids)
                    responses.append(response)
                    rewards.append(reward)
                    key = (problem['line_no'], response)
                    if reward > 0 and key not in seen_positive:
                        ultra_write_sft_line(positive_out, prompt, response, {
                            'reward': float(reward), 'predicted': predicted, 'from': 'grpo',
                            'line_no': problem['line_no']})
                        seen_positive.add(key)
                reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEV)
                reward_mean = reward_tensor.mean()
                reward_std = reward_tensor.std(unbiased=False)
                if reward_std.item() < 1e-8:
                    log_and_print('[GRPO E%d %d/%d] reward=%.3f 全组同分，跳过更新'
                                  % (epoch + 1, index, total_problems, reward_mean.item()))
                    continue
                advantages = (reward_tensor - reward_mean) / (reward_std + 1e-8)
                old_log_probs = [ultra_completion_log_probs(model, prompt_ids, completion)
                                 for completion in completions]
                for _ in range(inner_steps):
                    for completion, old_log_prob, advantage in zip(completions, old_log_probs, advantages):
                        if not completion:
                            continue
                        adv = float(advantage.item())
                        eps = float(clip_epsilon)
                        beta = float(old_policy_penalty)

                        def _wf(j, p, _adv=adv, _old=old_log_prob, _eps=eps, _beta=beta):
                            logp_new = float(torch.log(p[completion[j]]).item())
                            delta = logp_new - float(_old[j].item())
                            ratio = math.exp(delta)
                            if (_adv > 0 and ratio <= 1.0 + _eps) or (_adv < 0 and ratio >= 1.0 - _eps):
                                w = _adv * ratio
                            else:
                                w = 0.0
                            return w - 2.0 * _beta * delta

                        model.pg_update(prompt_ids, completion, _wf, lr)
                        total_updates += 1
                log_and_print('[GRPO E%d %d/%d] rewards=%s updates=%d'
                              % (epoch + 1, index, total_problems, rewards, total_updates))
            ultra_save_model_only(model, '%s_epoch%d.pt' % (checkpoint_prefix, epoch + 1), epoch + 1)
    log_and_print('GRPO 完成：共更新 %d 次；正确 rollout 已输出至 %s' % (total_updates, positive_output_path))


def ultra_rule_rl_train(model, tokenizer, problem_path: str,
                        positive_output_path: str, checkpoint_prefix: str,
                        epochs: int, max_new_tokens: int, lr: float,
                        baseline_momentum: float):
    if os.path.abspath(problem_path) == os.path.abspath(positive_output_path):
        raise ValueError('基础 RL 正确样本输出文件不能覆盖输入题库。')
    total_problems = ultra_count_problems(problem_path)
    if total_problems == 0:
        log_and_print('题库没有可用题目。')
        return
    baseline = 0.5
    total_updates = 0
    seen_positive = set()
    with open(positive_output_path, 'w', encoding='utf-8') as positive_out:
        for epoch in range(epochs):
            for index, problem in enumerate(ultra_iter_problems_shuffled(problem_path, buffer_size=10000, seed=epoch), 1):
                prompt = ultra_make_prompt(problem)
                prompt_ids = ultra_prompt_ids(tokenizer, prompt)
                completion_ids, response = ultra_generate_one(model, tokenizer, prompt_ids, max_new_tokens)
                reward, predicted = ultra_rule_reward(response, problem)
                advantage = reward - baseline
                baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * reward
                if reward > 0:
                    key = (problem['line_no'], response)
                    if key not in seen_positive:
                        ultra_write_sft_line(positive_out, prompt, response)
                        seen_positive.add(key)
                if not completion_ids or abs(advantage) < 1e-8:
                    log_and_print('[RL E%d %d/%d] reward=%.1f baseline=%.3f 跳过更新'
                                  % (epoch + 1, index, total_problems, reward, baseline))
                    continue

                def _wf(j, p, _adv=advantage):
                    return _adv

                model.pg_update(prompt_ids, completion_ids, _wf, lr)
                total_updates += 1
                log_and_print('[RL E%d %d/%d] reward=%.1f baseline=%.3f updates=%d'
                              % (epoch + 1, index, total_problems, reward, baseline, total_updates))
            ultra_save_model_only(model, '%s_epoch%d.pt' % (checkpoint_prefix, epoch + 1), epoch + 1)
    log_and_print('基础规则奖励 RL 完成：共更新 %d 次；正确 rollout 已输出至 %s' % (total_updates, positive_output_path))


def ultra_rejection_sampling_menu(model, tokenizer):
    problem_path = ultra_select_problem_file()
    if problem_path is None:
        return
    output_path = input('正确回答 SFT 输出文件 [ultra_rejection_sft.jsonl]: ').strip() or 'ultra_rejection_sft.jsonl'
    samples = ultra_read_int('每题采样回答数', ULTRA_RS_SAMPLES_PER_QUESTION)
    max_new_tokens = ultra_read_int('每个回答最多生成 token 数', ULTRA_RS_MAX_NEW_TOKENS)
    ultra_rejection_sampling(model, tokenizer, problem_path, output_path, samples, max_new_tokens)


def ultra_rule_rl_menu(model, tokenizer):
    problem_path = ultra_select_problem_file()
    if problem_path is None:
        return
    positive_output_path = input('正确 rollout 的 SFT 输出文件 [ultra_rl_positive.jsonl]: ').strip() or 'ultra_rl_positive.jsonl'
    checkpoint_prefix = input('基础 RL 模型 checkpoint 前缀 [ultra_rl]: ').strip() or 'ultra_rl'
    epochs = ultra_read_int('基础 RL epoch 数', ULTRA_RL_EPOCHS)
    max_new_tokens = ultra_read_int('每个回答最多生成 token 数', ULTRA_RL_MAX_NEW_TOKENS)
    lr = ultra_read_float('基础 RL 学习率', ULTRA_RL_LEARNING_RATE)
    baseline_momentum = ultra_read_float('奖励移动平均基线动量', ULTRA_RL_BASELINE_MOMENTUM)
    if baseline_momentum >= 1.0:
        log_and_print('基线动量必须小于 1.0。')
        return
    ultra_rule_rl_train(model, tokenizer, problem_path, positive_output_path, checkpoint_prefix,
                        epochs, max_new_tokens, lr, baseline_momentum)


def ultra_grpo_menu(model, tokenizer):
    problem_path = ultra_select_problem_file()
    if problem_path is None:
        return
    positive_output_path = input('正确 rollout 的 SFT 输出文件 [ultra_grpo_positive.jsonl]: ').strip() or 'ultra_grpo_positive.jsonl'
    checkpoint_prefix = input('GRPO 模型 checkpoint 前缀 [ultra_grpo]: ').strip() or 'ultra_grpo'
    epochs = ultra_read_int('GRPO epoch 数', ULTRA_GRPO_EPOCHS)
    group_size = ultra_read_int('每题同组采样回答数', ULTRA_GRPO_GROUP_SIZE, minimum=2)
    max_new_tokens = ultra_read_int('每个回答最多生成 token 数', ULTRA_GRPO_MAX_NEW_TOKENS)
    lr = ultra_read_float('GRPO 学习率', ULTRA_GRPO_LEARNING_RATE)
    inner_steps = ultra_read_int('每组参数更新次数', ULTRA_GRPO_INNER_STEPS)
    clip_epsilon = ultra_read_float('策略变化裁剪范围', ULTRA_GRPO_CLIP_EPSILON)
    old_policy_penalty = ultra_read_float('旧策略偏离惩罚系数', ULTRA_GRPO_OLD_POLICY_PENALTY)
    ultra_grpo_train(model, tokenizer, problem_path, positive_output_path, checkpoint_prefix,
                     epochs, group_size, max_new_tokens, lr, inner_steps,
                     clip_epsilon, old_policy_penalty)


# ============================================================
# P4：数据预处理 —— 超长对话切分（多轮切单轮 / 单轮超长丢弃）
# ============================================================
def p4_segment_dataset(input_path, output_path, tokenizer, threshold):
    """P4: 超长对话切分（规则与原版一致：不截断、不重训词表）。"""
    stats = {
        'total_lines': 0,
        'kept_samples': 0,
        'dropped_pairs': 0,
        'dropped_lines': 0,
        'kept_lines': 0,
    }

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats['total_lines'] += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            try:
                convs = _messages_from_record(data)
            except Exception:
                continue
            if not convs:
                continue

            # 开头的连续 system 段始终保留（与训练路径语义一致）
            head = []
            head_tokens = 0
            start = 0
            while start < len(convs) and convs[start].get('role') == 'system':
                m = convs[start]
                ids = tokenizer.encode(str(m.get('content') or ''))
                head.append(m)
                head_tokens += len(ids) + 2
                start += 1

            # 按 (user, assistant) 对切分
            pairs = []
            i = 0
            while i < len(convs) - 1:
                if convs[i].get('role') == 'user' and convs[i + 1].get('role') == 'assistant':
                    pairs.append((convs[i], convs[i + 1]))
                    i += 2
                else:
                    i += 1

            if not pairs:
                continue

            line_kept = 0
            for pair in pairs:
                pair_tokens = head_tokens
                for turn in pair:
                    content = turn.get('content', '')
                    ids = tokenizer.encode(content)
                    pair_tokens += len(ids) + 2      # 近似：role token + end token 开销
                    if TRAIN_ON_REASONING and turn.get('role') == 'assistant':
                        rc = turn.get('reasoning_content')
                        if rc:
                            pair_tokens += len(tokenizer.encode(str(rc)))
                    if turn.get('role') == 'assistant':
                        tc = _render_tool_calls(turn.get('tool_calls'))
                        if tc:
                            pair_tokens += len(tokenizer.encode(tc))
                pair_tokens += 1

                if pair_tokens <= threshold:
                    record = {'messages': head + list(pair)}
                    fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stats['kept_samples'] += 1
                    line_kept += 1
                else:
                    stats['dropped_pairs'] += 1

            if line_kept > 0:
                stats['kept_lines'] += 1
            else:
                stats['dropped_lines'] += 1

    log_and_print('=== P4 数据切分统计 ===')
    log_and_print('原行数: %d' % stats['total_lines'])
    log_and_print('保留样本数: %d' % stats['kept_samples'])
    log_and_print('保留的原行数: %d' % stats['kept_lines'])
    log_and_print('完全丢弃的行数: %d' % stats['dropped_lines'])
    log_and_print('丢弃的对话对数: %d' % stats['dropped_pairs'])
    log_and_print('输出文件: %s' % output_path)
    log_and_print('阈值: %d tokens' % threshold)
    return stats


def p4_segment_dataset_menu():
    """主菜单 11：数据预处理，切分超长对话。"""
    input_path = input('输入 jsonl 文件（默认 dialogues.jsonl）: ').strip() or 'dialogues.jsonl'
    if not os.path.exists(input_path):
        log_and_print('文件不存在: %s' % input_path)
        return

    default_out = input_path.replace('.jsonl', '_seg.jsonl')
    output_path = input('输出 jsonl 文件（默认 %s）: ' % default_out).strip() or default_out
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        log_and_print('输入输出不能是同一文件（避免覆盖）')
        return

    try:
        threshold_str = input('单样本 token 上限（默认 %d）: ' % P4_SEG_MAX_TOKENS).strip()
        threshold = int(threshold_str) if threshold_str else P4_SEG_MAX_TOKENS
    except ValueError:
        threshold = P4_SEG_MAX_TOKENS

    tokenizer, _path = _load_any_tokenizer()
    if tokenizer is None:
        log_and_print('未找到可用词表（%s），无法切分' % ' / '.join(TOKENIZER_FILES))
        return

    p4_segment_dataset(input_path, output_path, tokenizer, threshold)


# ============================================================
# 词表加载（原版流程：依次尝试 TOKENIZER_FILES；全无则从数据训练新词表）
# ============================================================
def _load_any_tokenizer():
    for cand in TOKENIZER_FILES:
        base = os.path.splitext(cand)[0]
        if os.path.exists(cand) or os.path.exists(base + '_tokenizer.json'):
            try:
                tk = Vocab()
                tk.load(cand)
                log_and_print('成功加载已有分词器：%s（词表大小 %d）' % (cand, tk.vocab_size))
                return tk, cand
            except Exception as e:
                log_and_print('分词器 %s 加载失败：%s' % (cand, e))
    return None, None


# ============================================================
# 主菜单（11 项与原版一致；内核已替换为 BioELM）
# ============================================================
def main():
    global TEMPERATURE, TOP_P
    init_log_file()
    tokenizer, _tk_path = _load_any_tokenizer()

    if tokenizer is None:
        log_and_print('准备训练新分词器。正在统计数据集字符规模...')
        try:
            with open('dialogues.jsonl', 'r', encoding='utf-8') as f:
                all_text = []
                char_set = set()
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    convs = _messages_from_record(data)
                    for turn in convs:
                        content = turn.get('content', '')
                        normalized = normalize_text(content)
                        all_text.append(normalized)
                        for ch in normalized:
                            char_set.add(ch)
                unique_chars = len(char_set)
                special_count = len(SPECIAL_TOKENS_LIST)
                required_min = unique_chars + special_count
                log_and_print('数据集包含 %d 个不同字符。若使用字符级词表至少需要词汇量 %d。' % (unique_chars, required_min))
                log_and_print('当前配置 VOCAB_SIZE = %d。' % VOCAB_SIZE)
                if VOCAB_SIZE < required_min:
                    log_and_print('警告：当前 VOCAB_SIZE 小于字符级最小需求，部分字符将被映射为 UNK。建议增大 VOCAB_SIZE 配置后重新运行。')
                tokenizer = Vocab()
                tokenizer.train(all_text, vocab_size=VOCAB_SIZE, verbose=True)
                tokenizer.save('tokenizer_incremental.bin')
                log_and_print('分词器训练完成并已保存。')
        except FileNotFoundError:
            log_and_print('未找到 dialogues.jsonl，扫描 .txt 文件用于分词器训练...')
            all_text = []
            char_set = set()
            for f in sorted(os.listdir('.')):
                if f.startswith('.'):
                    continue
                if f.endswith('.txt'):
                    try:
                        with open(f, 'r', encoding='utf-8') as tf:
                            for line in tf:
                                line = line.strip()
                                if line:
                                    normalized = normalize_text(line)
                                    all_text.append(normalized)
                                    for ch in normalized:
                                        char_set.add(ch)
                        log_and_print('  已读取 %s' % f)
                    except Exception as e:
                        log_and_print('  跳过 %s: %s' % (f, e))
            tokenizer = Vocab()
            if all_text:
                unique_chars = len(char_set)
                special_count = len(SPECIAL_TOKENS_LIST)
                required_min = unique_chars + special_count
                log_and_print('文本文件包含 %d 个不同字符。若使用字符级词表至少需要词汇量 %d。' % (unique_chars, required_min))
                log_and_print('当前配置 VOCAB_SIZE = %d。' % VOCAB_SIZE)
                if VOCAB_SIZE < required_min:
                    log_and_print('警告：当前 VOCAB_SIZE 小于字符级最小需求，部分字符将被映射为 UNK。建议增大 VOCAB_SIZE 配置后重新运行。')
                tokenizer.train(all_text, vocab_size=VOCAB_SIZE, verbose=True)
            else:
                log_and_print('未找到任何可用的 .txt 文件，将使用空文本训练分词器（仅保留特殊token）。')
                tokenizer.train([''], vocab_size=VOCAB_SIZE, verbose=False)
            tokenizer.save('tokenizer_incremental.bin')

    model = BioELM(tokenizer.vocab_size)
    log_and_print('BioELM 架构：选择性门控对角 SSM + 精确在线 RTRL + 三因子局部更新（零 autograd）；'
                  'H=%d d=%d m=%d 参数量=%d' % (model.h, model.d, model.m, model.n_params()))

    model_loaded = False
    if os.path.exists('brain.pt'):
        log_and_print('检测到 brain.pt，训练或对话时按需加载（避免启动即占用冗余显存）。')
    else:
        log_and_print('未找到 brain.pt，使用随机初始化模型。')

    def ensure_model_loaded() -> bool:
        """首次用到模型时，若 brain.pt 存在则加载（checkpoint 临时放内存，写入后立即释放）。"""
        nonlocal model_loaded
        if model_loaded:
            return True
        if not os.path.exists('brain.pt'):
            return False
        try:
            load_brain_pt_into(model)
        except Exception as e:
            log_and_print('加载 brain.pt 失败: %s。为避免随机初始化误训练，程序退出。' % e)
            close_log_file()
            raise SystemExit(1)
        model_loaded = True
        log_and_print('已加载 brain.pt 模型权重。')
        return True

    while True:
        print('\n=== BioELM 训练与对话系统菜单 ===')
        print('1. 训练模型（继续训练）')
        print('2. 对话模式')
        print('3. 保存模型')
        print('4. 加载模型')
        print('5. 调整生成参数')
        print('6. 查看模型统计信息')
        print('7. 规则验证拒绝采样：生成正确回答 SFT 数据')
        print('8. 基础规则奖励 RL：单回答试错训练')
        print('9. GRPO 规则强化学习：同题分组相对优化')
        print('10. 退出')
        print('11. 数据预处理：超长对话切分（多轮切单轮 / 单轮超长丢弃）')
        choice = input('请选择: ').strip()

        if choice == '1':
            data_path, file_ext = select_training_file()
            if data_path is None:
                continue
            if not ensure_model_loaded():
                log_and_print('未找到 brain.pt，使用随机初始化模型从头训练。')
            if input('开始训练？(y/n): ').lower() == 'y':
                train_from_scratch(model, tokenizer, data_path,
                                   BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                                   GRAD_CLIP_NORM, LOG_INTERVAL,
                                   accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                                   file_type=file_ext)
        elif choice == '2':
            ensure_model_loaded()
            chat_mode = input('对话格式 (jsonl) 还是纯文本续写 (txt)？[jsonl]: ').strip().lower()
            if chat_mode not in ('jsonl', 'txt'):
                chat_mode = 'jsonl'
            chat_loop(model, tokenizer, mode=chat_mode)
        elif choice == '3':
            try:
                save_dict = {
                    'model_state_dict': model_state_dict(model),
                    'epoch': 0,
                    'bio_meta': {'h': model.h, 'd': model.d, 'm': model.m,
                                 'vocab': model.vocab, 'steps': model.steps},
                }
                torch.save(save_dict, 'brain.pt')
                log_and_print('Model saved to brain.pt.')
                model_loaded = True
            except Exception as e:
                log_and_print('保存失败: %s' % e)
        elif choice == '4':
            try:
                if load_brain_pt_into(model):
                    log_and_print('模型加载成功。')
                    model_loaded = True
                else:
                    log_and_print('未找到 brain.pt。')
            except Exception as e:
                log_and_print('Model load failed: %s. Stopping to avoid accidental random initialization.' % e)
                close_log_file()
                return
        elif choice == '5':
            try:
                TEMPERATURE = float(input('temperature (0.1-1.5) [当前 %.2f]: ' % TEMPERATURE) or TEMPERATURE)
                TOP_P = float(input('top_p (0.0-1.0) [当前 %.2f]: ' % TOP_P) or TOP_P)
            except ValueError:
                log_and_print('输入无效，保持原值')
        elif choice == '6':
            total_params = model.n_params()
            log_and_print('可训练参数: %d / 总参数: %d（内核无冻结层概念，全部参数可训）'
                          % (total_params, total_params))
            log_and_print('词表大小: %d' % model.vocab)
            log_and_print('状态单元 H: %d | 感官维度 d: %d | 读出维度 m: %d | 元学习率 η: %.2e | 权重衰减: %.1e'
                          % (model.h, model.d, model.m, model.eta, model.wd))
            log_and_print('已训练步数: %d | 设备: %s' % (model.steps, str(DEV)))
            if model.adapt:
                log_and_print('[O11] 自适应：因子 s=%.3f（界 [%.3f, %.1f]）| 反馈窗 τ=%d 步 | 损失慢迹=%.4f | ρ=%.0e'
                              % (float(model._s), CTRL_S_MIN, CTRL_S_MAX, model._win_len,
                                 float(model._loss_ema), TRUST_RHO))
            log_and_print('架构：选择性门控对角 SSM + 精确在线 RTRL + 三因子局部更新（零 autograd）')
            log_and_print('更新方式：学习信号 e ⊗ 资格迹 → 元 Adam（RMS 归一，时间步偏差校正）；'
                          '写入门 g=σ(xWg+bg)，选择性遗忘 d=σ(xWd+bd)，读出 sem=σ(sWp+bp)')
        elif choice == '7':
            ensure_model_loaded()
            ultra_rejection_sampling_menu(model, tokenizer)
        elif choice == '8':
            ensure_model_loaded()
            ultra_rule_rl_menu(model, tokenizer)
        elif choice == '9':
            ensure_model_loaded()
            ultra_grpo_menu(model, tokenizer)
        elif choice == '11':
            p4_segment_dataset_menu()
        elif choice == '10':
            break
        else:
            log_and_print('无效选择，请重新输入')
    close_log_file()


if __name__ == '__main__':
    main()
