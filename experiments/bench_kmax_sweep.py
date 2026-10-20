"""random 模式 KMAX 扫描（GPU, h=512, 固定 20k 步）：KMAX ∈ {2,3,4,6}，anytime@K=1..6。"""
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


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LOG_FILE = os.path.join(os.environ.get('TEMP', '.'), 'kmax_sweep_%s.log' % name)
    mod.init_log_file()
    return mod


EA = load('bioelm_elastic_1.1.0')
tok, _ = EA._load_any_tokenizer()
DATA = os.path.join(BASE, 'dsr1_v5_0001.jsonl')
tr_ids, tr_mask, va_ids, va_mask = EA.build_corpus(tok, DATA)
vs, vt, vm = EA.to_streams(va_ids, va_mask, NV)


def run(kmax):
    EA.LOOP_TRAIN_MODE = 'random'
    EA.LOOP_K_MAX = kmax
    model = EA.BioELM(tok.vocab_size, h=512, d=128, m=128)
    model.loop_mode = 'random'
    model.loop_k_max = kmax
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
    for k in (1, 2, 3, 4, 5, 6):
        model.loop_k = k
        scores.append(float(model.eval_masked(vs, vt, vm)))
    print('[kmax%d] %d steps %.1fs (%.1f st/s) | %s' % (kmax, steps, dt, steps / dt,
          ' '.join('K%d=%.4f' % (k, c) for k, c in zip((1, 2, 3, 4, 5, 6), scores))), flush=True)
    del model
    torch.cuda.empty_cache()


for km in (2, 3, 6):
    run(km)
print('DONE')
