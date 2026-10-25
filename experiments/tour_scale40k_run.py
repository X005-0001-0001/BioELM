"""CPU: training-step scaling under 1.1.9 recipe (SRMS + EMA + LSM).
128 streams x 40k x 2 seeds, checkpoints 20k/30k/40k.
Ref: 20k (same protocol) = 3.2166."""
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

CP = (20000, 30000, 40000)
CFG = {'K': 1, 'TAU': 1.0, 'h': 256, 'd': 192, 'm': 192,
       'streams': 128, 'steps': 40000, 'checkpoints': CP,
       'EMA': 0.999, 'LSM': 0.05, 'SRMS': 1}

res = {c: [] for c in CP}
for seed in (43, 44):
    out = L.run(dict(CFG, seed=seed))
    print('  seed=%d ' % seed + ' '.join('%d:%.4f' % (c, out[c]) for c in CP), flush=True)
    for c in CP:
        res[c].append(out[c])
print('scale40k ' + ' '.join('%.4f+-%.3f' % (
    statistics.mean(res[c]), statistics.stdev(res[c])) for c in CP), flush=True)
print('DONE')
