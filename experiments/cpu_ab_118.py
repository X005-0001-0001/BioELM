"""CPU product A/B: elite 1.1.7 (EMA) vs 1.1.8 (EMA + LSM 0.05).
20k steps, 64 streams, 2 seeds, full val, eval via _eval_streams."""
import os
os.environ['BIO_DEV'] = 'cpu'
os.environ['BIO_LSM'] = '0.05'
import sys
import math
import time
import statistics
import importlib.util
import torch

BASE = r'D:\开源系列\BioELM'
sys.path.insert(0, BASE)
os.chdir(BASE)
DEV = torch.device('cpu')
NS, NV = 64, 64
STEPS = 20000
DATA = os.path.join(BASE, 'dsr1_v5_0001.jsonl')


def load(fn, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LOG_FILE = os.path.join(os.environ.get('TEMP', '.'), name + '.log')
    mod.init_log_file()
    return mod


M7 = load('bioelm_elite_1.1.7.py', 'p117')
M8 = load('bioelm_elite_1.1.8.py', 'p118')
tok, _ = M7._load_any_tokenizer()
tr_ids, tr_mask, va_ids, va_mask = M7.build_corpus(tok, DATA)


def train_eval(mod, label, seed):
    torch.manual_seed(seed)
    model = mod.BioELM(tok.vocab_size, seed=seed)
    st = model.init_stream_state(NS)
    t0 = time.time()
    done = 0
    rot = 0
    while done < STEPS:
        s = torch.tensor(tr_ids[rot:rot + NS * 400 + 1])
        bt = s[:-1].view(NS, 400).t().contiguous()
        bg = s[1:].view(NS, 400).t().contiguous()
        bm = torch.tensor(tr_mask[rot + 1:rot + NS * 400 + 1],
                          dtype=torch.float32).view(NS, 400).t().contiguous()
        for t in range(bt.shape[0]):
            model.step(bt[t], bg[t], st, train=True, mask=bm[t])
            done += 1
            if done >= STEPS:
                break
        rot = rot + NS * 400 + 1
        if rot + NS * 400 + 2 >= len(tr_ids):
            rot = 0
    dt = time.time() - t0
    ce = float(mod._eval_streams(model, va_ids, va_mask, ns=NV))
    print('[%s seed=%d] %d steps %.1fs (%.1f st/s) valCE=%.4f ppl=%.2f' % (
        label, seed, done, dt, done / dt, ce, math.exp(ce)), flush=True)
    return ce


res = {43: {}, 44: {}}
for seed in (43, 44):
    res[seed]['1.1.7'] = train_eval(M7, '1.1.7', seed)
    res[seed]['1.1.8'] = train_eval(M8, '1.1.8', seed)
for label in ('1.1.7', '1.1.8'):
    vals = [res[s][label] for s in (43, 44)]
    print('%s mean=%.4f pm=%.4f' % (label, statistics.mean(vals), statistics.stdev(vals)))
print('DONE')
