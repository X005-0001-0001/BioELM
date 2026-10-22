"""CPU: SRMS 下的超参重扫 + 输入 RMSNorm。128 流 x 6k x 2 种子。
基线（SRMS=1, eta=2e-3, wd=1e-4）= 3.4207."""
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
CP = (3000, 6000)
BASE_CFG = {'K': 1, 'TAU': 1.0, 'h': 256, 'd': 192, 'm': 192,
            'streams': 128, 'steps': 6000, 'checkpoints': CP,
            'EMA': 0.999, 'LSM': 0.05, 'SRMS': 1}

GRID = [
    ('eta3e-3', {'ETA': 3e-3}),
    ('wd3e-4', {'wd': 3e-4}),
    ('xnrm', {'XNRM': 1}),
]

for name, over in GRID:
    res = {c: [] for c in CP}
    for seed in SEEDS:
        out = L.run(dict(BASE_CFG, seed=seed, **over))
        for c in CP:
            res[c].append(out[c])
    print(name + ' ' + ' '.join('%.4f+-%.3f' % (
        statistics.mean(res[c]), statistics.stdev(res[c])) for c in CP), flush=True)
print('DONE')
