"""Round 34 (GPU): context-length extrapolation of the released config.
Train 20k (H256/D192/M192, K1, 1024 streams), eval at TV in {200, 512, 1024, 2048, 4096}.
Training chunk is 400 — measures what happens beyond it."""
import os
os.environ.setdefault('BIO_DEV', 'cuda')
import sys
import importlib.util
import torch

BASE = r'D:\开源系列\BioELM'
sys.path.insert(0, BASE)
os.chdir(BASE)

spec = importlib.util.spec_from_file_location('lab', os.path.join(BASE, 'experiments', 'struct_lab.py'))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

SEED = 43
L.CFG.update({'K': 1, 'TAU': 1.0, 'h': 256, 'd': 192, 'm': 192, 'seed': SEED,
              'streams': 1024, 'steps': 20000, 'checkpoints': ()})
L.K, L.SEED = 1, SEED
L.H, L.D, L.M = 256, 192, 192
net = L.Net()
st = net.init_state(1024)
rot = 0
done = 0
while done < 20000:
    bt, bg, bm, bg2 = L.to_batches(L.tr_ids, L.tr_mask, 1024, L.CHUNK, rot)
    for s in range(bt.shape[0]):
        net.step(bt[s], bg[s], st, train=True, mask=bm[s])
        done += 1
        if done >= 20000:
            break
    rot = rot + 1024 * L.CHUNK + 1
    if rot + 1024 * L.CHUNK + 2 >= len(L.tr_ids):
        rot = 0

import math
for TV, NV in ((200, 128), (512, 64), (1024, 32), (2048, 16), (4096, 8)):
    ce = L.evaluate(net, L.va_ids, L.va_mask, NV, TV)
    print('TV=%4d NV=%3d valCE=%.4f ppl=%.2f' % (TV, NV, ce, math.exp(ce)), flush=True)
print('DONE')
