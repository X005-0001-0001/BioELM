"""CPU: LSM epsilon sweep at 20k under SRMS. 128 streams x 2 seeds.
Ref: eps=0.05, SRMS @20k = 3.2166."""
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

BASE_CFG = {'K': 1, 'TAU': 1.0, 'h': 256, 'd': 192, 'm': 192, 'streams': 128,
            'EMA': 0.999, 'SRMS': 1, 'steps': 20000, 'checkpoints': (10000, 20000)}

for eps in (0.10, 0.15):
    res = {c: [] for c in BASE_CFG['checkpoints']}
    for seed in (43, 44):
        out = L.run(dict(BASE_CFG, LSM=eps, seed=seed))
        for c in BASE_CFG['checkpoints']:
            res[c].append(out[c])
    print('lsm%.2f ' % eps + ' '.join('%.4f+-%.3f' % (
        statistics.mean(res[c]), statistics.stdev(res[c])) for c in BASE_CFG['checkpoints']), flush=True)
print('DONE')
