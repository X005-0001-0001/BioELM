"""consist 预热 → random 主训练（GPU, h=192, 全语料, 20k 步）：
对照：纯 random 20k：K1 3.2995 K2 3.1400 K3 3.1146 K4 3.1385
      纯 consist 20k：K1 3.4184 K2 3.3149 K3 3.2632 K4 3.2567
      纯 consist 3k： K1 3.5597 K2 3.4849 K3 3.4571 K4 3.4605
      纯 random  3k： K1 3.6847 K2 3.5075 K3 3.4816 K4 3.4985
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


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LOG_FILE = os.path.join(os.environ.get('TEMP', '.'), 'warmup_%s.log' % name)
    mod.init_log_file()
    return mod


EA = load('bioelm_elastic_1.1.0')
tok, _ = EA._load_any_tokenizer()
DATA = os.path.join(BASE, 'dsr1_v5_0001.jsonl')
tr_ids, tr_mask, va_ids, va_mask = EA.build_corpus(tok, DATA)
vs, vt, vm = EA.to_streams(va_ids, va_mask, NV)


def run(label, warmup_steps):
    EA.LOOP_K_MAX = 4
    model = EA.BioELM(tok.vocab_size, h=H, d=64, m=64)
    model.loop_mode = 'consist' if warmup_steps > 0 else 'random'
    model.loop_k_max = 4
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
            if steps == warmup_steps:
                model.loop_mode = 'random'
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
    print('[%s] | %s' % (label,
          ' '.join('K%d=%.4f' % (k, c) for k, c in zip((1, 2, 3, 4), scores))), flush=True)
    del model
    torch.cuda.empty_cache()


run('warm3k-consist', 3000)
run('warm5k-consist', 5000)
print('DONE')
