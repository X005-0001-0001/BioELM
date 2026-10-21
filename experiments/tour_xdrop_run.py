"""CPU screen: input feature dropout with EMA (128 streams x 6k x 2 seeds)."""
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
            'streams': 128, 'steps': 6000, 'checkpoints': CP, 'EMA': 0.999}

for name, p in (('xdrop0.1', 0.1), ('xdrop0.2', 0.2)):
    res = {c: [] for c in CP}
    for seed in SEEDS:
        out = L.run(dict(BASE_CFG, seed=seed, XDROP=p))
        for c in CP:
            res[c].append(out[c])
    print('%-9s ' % name + ' '.join('%.4f+-%.3f' % (
        statistics.mean(res[c]), statistics.stdev(res[c])) for c in CP), flush=True)
print('DONE')
