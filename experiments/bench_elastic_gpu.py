"""elastic GPU 大尺度对照台：固定步数下 random / fixed / consist 的 anytime 质量与每步成本。

用法（仓库根目录，默认 GPU）：
    python experiments/bench_elastic_gpu.py
    BIO_DEV=cpu BIO_ELASTIC_SUBSET=500 BIO_ELASTIC_STEPS=3000 python experiments/bench_elastic_gpu.py
环境变量：BIO_ELASTIC_STEPS（默认 20000）、BIO_ELASTIC_H（默认 512）、BIO_ELASTIC_NS（默认 512）、
BIO_ELASTIC_SUBSET（>0 时自动截取子集，CPU 快速验证用）。
结论（GPU h=512 20k 步，见 elastic 1.1.0 发布说明）：random 每步与每墙钟全档最优（默认），
consist 为小尺度方法（h=192 CPU 更优、h=512 降质），fixed 深预算略优但浅预算崩溃。
"""
import os
import sys
import time
import random
import importlib.util
import tempfile

os.environ.setdefault('BIO_DEV', 'cuda')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

import torch
DEV = torch.device(os.environ.get('BIO_DEV', 'cuda'))
STEPS = int(os.environ.get('BIO_ELASTIC_STEPS', '20000'))
NS = int(os.environ.get('BIO_ELASTIC_NS', '512'))
H = int(os.environ.get('BIO_ELASTIC_H', '512'))
D, M = 128, 128
NV = 128


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LOG_FILE = os.path.join(tempfile.gettempdir(), 'bench_elastic_gpu_%s.log' % name)
    mod.init_log_file()
    return mod


EA = load('bioelm_elastic_1.1.0')
tok, _ = EA._load_any_tokenizer()
DATA = os.environ.get('BIO_ELASTIC_DATA', os.path.join(BASE, 'dsr1_v5_0001.jsonl'))
SUBSET = int(os.environ.get('BIO_ELASTIC_SUBSET', '0'))
if SUBSET > 0:
    sub = os.path.join(tempfile.gettempdir(), 'bench_elastic_gpu_sub.jsonl')
    with open(DATA, encoding='utf-8') as f, open(sub, 'w', encoding='utf-8') as g:
        for i, line in enumerate(f):
            if i >= SUBSET:
                break
            g.write(line)
    DATA = sub

tr_ids, tr_mask, va_ids, va_mask = EA.build_corpus(tok, DATA)
vs, vt, vm = EA.to_streams(va_ids, va_mask, NV)


def run(label, mode, lam=None):
    EA.LOOP_TRAIN_MODE = mode
    if lam is not None:
        EA.LOOP_CONSIST_LAMBDA = lam
    model = EA.BioELM(tok.vocab_size, h=H, d=D, m=M)
    model.loop_mode = mode
    st = model.init_stream_state(NS)
    t0 = time.time()
    steps = 0
    epoch = 0
    done = False
    while not done:
        rotate = 0 if epoch == 0 else random.randrange(0, max(1, len(tr_ids)))
        st_s, tg_s, mk_s = EA.to_streams(tr_ids, tr_mask, NS, rotate=rotate)
        st_t = st_s.t().contiguous().to(DEV)
        tg_t = tg_s.t().contiguous().to(DEV)
        mk_t = mk_s.t().contiguous().to(DEV)
        for s in range(st_t.shape[0]):
            model.step(st_t[s], tg_t[s], st, train=True, mask=mk_t[s])
            steps += 1
            if steps >= STEPS:
                done = True
                break
        epoch += 1
    dt = time.time() - t0
    scores = []
    for k in (1, 2, 3, 4):
        model.loop_k = k
        scores.append(float(model.eval_masked(vs, vt, vm)))
    print('[%s] %d steps %.1fs (%.1f st/s) | %s' % (label, steps, dt, steps / dt,
          ' '.join('K%d=%.4f' % (k, c) for k, c in zip((1, 2, 3, 4), scores))), flush=True)
    del model
    if DEV.type == 'cuda':
        torch.cuda.empty_cache()


run('random', 'random')
run('fixedK4', 'fixed')
run('consist', 'consist')
print('DONE')
