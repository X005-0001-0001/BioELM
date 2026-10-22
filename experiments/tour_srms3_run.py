"""CPU: (a) eta=3e-3 with SRMS @6k  (b) XNRM @20k.  128 流 x 2 种子.
Refs: SRMS @6k=3.4207, @20k=3.2166."""
import os
os.environ['BIO_DEV'] = 'cpu'
import sys
import statistics
import importlib.util

BASE = r'D:\开源系列\BioELM'
sys.path.insert(0, BASE)
os.chdir(BASE)

spec = importlib.util.spec_from_file_location('lab', os.path.join(BASE, 'experiments', 'struct_lab.py'))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
L.torch.set_num_threads(4)

SEEDS = (43, 44)
BASE_CFG = {'K': 1, 'TAU': 1.0, 'h': 256, 'd': 192, 'm': 192,
            'streams': 128, 'EMA': 0.999, 'LSM': 0.05, 'SRMS': 1}

JOBS = [
    ('eta3e-3', dict(BASE_CFG, steps=6000, checkpoints=(3000, 6000), eta=3e-3)),
    ('xnrm20k', dict(BASE_CFG, steps=20000, checkpoints=(10000, 20000), XNRM=1)),
]

for name, cfgb in JOBS:
    CP = cfgb['checkpoints']
    res = {c: [] for c in CP}
    for seed in SEEDS:
        out = L.run(dict(cfgb, seed=seed))
        for c in CP:
            res[c].append(out[c])
    print(name + ' ' + ' '.join('%.4f+-%.3f' % (
        statistics.mean(res[c]), statistics.stdev(res[c])) for c in CP), flush=True)
print('DONE')
