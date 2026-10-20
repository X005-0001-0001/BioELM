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
STEPS = 20000


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LOG_FILE = os.path.join(os.environ.get('TEMP', '.'), 'k3_%s.log' % name)
    mod.init_log_file()
    return mod


A = load('bioelm_deep_1.0.1')
B = load('bioelm_deep_1.1.0')
tok, _ = A._load_any_tokenizer()
DATA = os.path.join(BASE, 'dsr1_v5_0001.jsonl')
tr_ids, tr_mask, va_ids, va_mask = A.build_corpus(tok, DATA)
vs, vt, vm = A.to_streams(va_ids, va_mask, NV)


def quality(mod, label, loop_k):
    mod.LOOP_K = loop_k
    model = mod.BioELM(tok.vocab_size, h=512, d=128, m=128)
    st = model.init_stream_state(NS)
    done = 0
    epoch = 0
    while done < STEPS:
        rotate = 0 if epoch == 0 else random.randrange(0, len(tr_ids))
        st_s, tg_s, mk_s = A.to_streams(tr_ids, tr_mask, NS, rotate=rotate)
        st_t = st_s.t().contiguous().to(DEV)
        tg_t = tg_s.t().contiguous().to(DEV)
        mk_t = mk_s.t().contiguous().to(DEV)
        for s in range(st_t.shape[0]):
            model.step(st_t[s], tg_t[s], st, train=True, mask=mk_t[s])
            done += 1
            if done >= STEPS:
                break
        epoch += 1
    ce = float(model.eval_masked(vs, vt, vm))
    print('[%s] 20k valCE=%.4f ppl=%.2f tc=%s' % (label, ce, math.exp(ce),
          getattr(model, 'loop_timecond', None)), flush=True)


quality(A, '101-K3', 3)
quality(B, '110-K3', 3)
print('DONE')
