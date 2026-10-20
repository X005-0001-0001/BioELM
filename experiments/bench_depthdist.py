"""random 模式的深度分布形状对照（GPU, h=192, 20k 步, KMAX=4）：
uniform（默认 U{1..4}，均值 2.5）| bimodal（1 与 4 各半，均值 2.5）|
geometric（P∝2^-k，均值约 1.5）| deep-biased（P∝k，均值 3.0）。
同均值对照（uniform vs bimodal）检验"形状"本身是否有用；两端偏置检验深度覆盖 vs 深预算质量。
"""
import os
import sys
import time
import random
import importlib.util

os.environ['BIO_DEV'] = 'cuda'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

import torch
DEV = torch.device('cuda')
STEPS = 20000
NS = 512
NV = 128
H = 192
KMAX = 4


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LOG_FILE = os.path.join(os.environ.get('TEMP', '.'), 'depthdist_%s.log' % name)
    mod.init_log_file()
    return mod


EA = load('bioelm_elastic_1.1.0')
tok, _ = EA._load_any_tokenizer()
DATA = os.path.join(REPO, 'dsr1_v5_0001.jsonl')
tr_ids, tr_mask, va_ids, va_mask = EA.build_corpus(tok, DATA)
vs, vt, vm = EA.to_streams(va_ids, va_mask, NV)


class DistRng:
    def __init__(self, kind, kmax, seed=43):
        self.kind = kind
        self.kmax = kmax
        self.r = random.Random(seed)

    def randint(self, a, b):
        k = self.kmax
        if self.kind == 'uniform':
            return self.r.randint(a, b)
        if self.kind == 'bimodal':
            return 1 if self.r.random() < 0.5 else k
        if self.kind == 'geom':
            w = [2.0 ** (-i) for i in range(1, k + 1)]
        elif self.kind == 'deep':
            w = [float(i) for i in range(1, k + 1)]
        else:
            return self.r.randint(a, b)
        s = sum(w)
        x = self.r.random() * s
        c = 0.0
        for i, wi in enumerate(w):
            c += wi
            if x <= c:
                return i + 1
        return k


def run(kind):
    EA.LOOP_TRAIN_MODE = 'random'
    EA.LOOP_K_MAX = KMAX
    model = EA.BioELM(tok.vocab_size, h=H, d=64, m=64)
    model.loop_mode = 'random'
    model.loop_k_max = KMAX
    model._loop_rng = DistRng(kind, KMAX)
    st = model.init_stream_state(NS)
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
    scores = []
    for k in (1, 2, 3, 4):
        model.loop_k = k
        scores.append(float(model.eval_masked(vs, vt, vm)))
    print('[%s] | %s' % (kind,
          ' '.join('K%d=%.4f' % (k, c) for k, c in zip((1, 2, 3, 4), scores))), flush=True)
    del model
    torch.cuda.empty_cache()


for kind in ('uniform', 'bimodal', 'geom', 'deep'):
    run(kind)
print('DONE')
