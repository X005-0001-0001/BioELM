"""CPU screen: weight decay fine sweep with EMA (128 streams x 6k x 2 seeds)."""
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

for name, wd in (('wd3e-4', 3.0e-4), ('wd3e-5', 3.0e-5)):
    res = {c: [] for c in CP}
    for seed in SEEDS:
        out = L.run(dict(BASE_CFG, seed=seed, wd=wd))
        for c in CP:
            res[c].append(out[c])
    print('%-8s ' % name + ' '.join('%.4f+-%.3f' % (
        statistics.mean(res[c]), statistics.stdev(res[c])) for c in CP), flush=True)
print('DONE')
