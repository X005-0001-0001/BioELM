"""CPU: state RMSNorm 20k validation (EMA + LSM). 128 streams x 2 seeds.
Baseline (Round 48, same protocol): 3.2973 +/- 0.002."""
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
CP = (10000, 20000)
BASE_CFG = {'K': 1, 'TAU': 1.0, 'h': 256, 'd': 192, 'm': 192,
            'streams': 128, 'steps': 20000, 'checkpoints': CP,
            'EMA': 0.999, 'LSM': 0.05, 'SRMS': 1}

res = {c: [] for c in CP}
for seed in SEEDS:
    out = L.run(dict(BASE_CFG, seed=seed))
    for c in CP:
        res[c].append(out[c])
print('srms20k ' + ' '.join('%.4f+-%.3f' % (
    statistics.mean(res[c]), statistics.stdev(res[c])) for c in CP), flush=True)
print('DONE')
