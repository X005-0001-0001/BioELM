"""CPU: learnable per-dim gain after state RMSNorm (SGAIN). 128 streams x 6k x 2 seeds.
Ref: SRMS baseline @6k = 3.4207."""
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

CP = (3000, 6000)
CFG = {'K': 1, 'TAU': 1.0, 'h': 256, 'd': 192, 'm': 192,
       'streams': 128, 'steps': 6000, 'checkpoints': CP,
       'EMA': 0.999, 'LSM': 0.05, 'SRMS': 1, 'SGAIN': 1}

res = {c: [] for c in CP}
for seed in (43, 44):
    out = L.run(dict(CFG, seed=seed))
    for c in CP:
        res[c].append(out[c])
    m = L.M
print('sgain ' + ' '.join('%.4f+-%.3f' % (
    statistics.mean(res[c]), statistics.stdev(res[c])) for c in CP), flush=True)
print('DONE')
