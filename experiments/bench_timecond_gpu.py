"""时间条件归因（GPU, h=512）：
A) 1.1.0 tc-off vs tc-on 各 60s 吞吐（隔离机制开销）
B) K=4 固定 20k 步：1.0.1 vs 1.1.0（检验 K≥3 时是否有益）
"""
import os
import sys
import time
import math
import random
import importlib.util

os.environ['BIO_DEV'] = 'cuda'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

import torch
DEV = torch.device('cuda')
NS = 512
NV = 128


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LOG_FILE = os.path.join(os.environ.get('TEMP', '.'), 'attr_%s.log' % name)
    mod.init_log_file()
    return mod


A = load('bioelm_deep_1.0.1')
B = load('bioelm_deep_1.1.0')
tok, _ = A._load_any_tokenizer()
DATA = os.path.join(BASE, 'dsr1_v5_0001.jsonl')
tr_ids, tr_mask, va_ids, va_mask = A.build_corpus(tok, DATA)
vs, vt, vm = A.to_streams(va_ids, va_mask, NV)


def streams(rot):
    st_s, tg_s, mk_s = A.to_streams(tr_ids, tr_mask, NS, rotate=rot)
    return (st_s.t().contiguous().to(DEV), tg_s.t().contiguous().to(DEV),
            mk_s.t().contiguous().to(DEV))


def throughput(mod, label, tc=None, loop_k=2, seconds=60):
    mod.LOOP_K = loop_k
    if tc is not None:
        mod.LOOP_TIMECOND = tc
    model = mod.BioELM(tok.vocab_size, h=512, d=128, m=128)
    st = model.init_stream_state(NS)
    t0 = time.time()
    steps = 0
    st_t, tg_t, mk_t = streams(0)
    T = st_t.shape[0]
    while time.time() - t0 < seconds:
        s = steps % T
        if s == 0 and steps:
            st_t, tg_t, mk_t = streams(random.randrange(0, len(tr_ids)))
        model.step(st_t[s], tg_t[s], st, train=True, mask=mk_t[s])
        steps += 1
    dt = time.time() - t0
    print('[%s] %.1f st/s (tc=%s)' % (label, steps / dt, getattr(model, 'loop_timecond', None)), flush=True)
    del model
    torch.cuda.empty_cache()


def quality(mod, label, loop_k, steps=20000):
    mod.LOOP_K = loop_k
    model = mod.BioELM(tok.vocab_size, h=512, d=128, m=128)
    st = model.init_stream_state(NS)
    done = 0
    epoch = 0
    while done < steps:
        st_t, tg_t, mk_t = streams(0 if epoch == 0 else random.randrange(0, len(tr_ids)))
        T = st_t.shape[0]
        for s in range(T):
            model.step(st_t[s], tg_t[s], st, train=True, mask=mk_t[s])
            done += 1
            if done >= steps:
                break
        epoch += 1
    ce = float(model.eval_masked(vs, vt, vm))
    print('[%s] 20k valCE=%.4f ppl=%.2f tc=%s' % (label, ce, math.exp(ce),
          getattr(model, 'loop_timecond', None)), flush=True)
    del model
    torch.cuda.empty_cache()


throughput(B, '110-tcOFF', tc=False)
throughput(B, '110-tcON', tc=True)
quality(A, '101-K4', 4)
quality(B, '110-K4', 4)
print('DONE')
