"""Round 36 (GPU): training-mechanism / dynamics scan at the released config.
H256/D192/M192, K1, 1024 streams x 20k x 3 seeds."""
import os
os.environ.setdefault('BIO_DEV', 'cuda')
import sys
import statistics
import importlib.util

BASE = r'D:\开源系列\BioELM'
sys.path.insert(0, BASE)
os.chdir(BASE)

spec = importlib.util.spec_from_file_location('lab', os.path.join(BASE, 'experiments', 'struct_lab.py'))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

SEEDS = (43, 44, 45)
CP = (10000, 20000)
BASE_CFG = {'K': 1, 'TAU': 1.0, 'h': 256, 'd': 192, 'm': 192,
            'streams': 1024, 'steps': 20000, 'checkpoints': CP}

VARIANTS = [
    ('base', {}),
    ('lam0.9', {'TRACE_LAM': 0.9}),
    ('lam0.99', {'TRACE_LAM': 0.99}),
    ('lrR2', {'LR_R': 2.0}),
    ('lrS0.5', {'LR_S': 0.5}),
    ('silu', {'RACT': 'silu'}),
    ('b1_0.5', {'b1': 0.5}),
]
for name, extra in VARIANTS:
    res = {c: [] for c in CP}
    for seed in SEEDS:
        cfg = dict(BASE_CFG, seed=seed, **extra)
        out = L.run(cfg)
        for c in CP:
            res[c].append(out[c])
    print('%-8s ' % name + ' '.join('%.4f+-%.3f' % (
        statistics.mean(res[c]), statistics.stdev(res[c])) for c in CP), flush=True)
print('DONE')
