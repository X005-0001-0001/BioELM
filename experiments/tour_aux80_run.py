"""Round 38b (GPU): aux persistence at 80k (AUX_R=128, lam=0.5) vs known baseline
2.8707/2.8405 @40k/80k. H256/D192/M192, K1, 1024 streams x 3 seeds."""
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

CP = (40000, 80000)
res = {c: [] for c in CP}
for seed in (43, 44, 45):
    out = L.run({'K': 1, 'TAU': 1.0, 'h': 256, 'd': 192, 'm': 192, 'seed': seed,
                 'AUX': 1, 'AUX_LAM': 0.5, 'AUX_R': 128,
                 'streams': 1024, 'steps': 80000, 'checkpoints': CP})
    for c in CP:
        res[c].append(out[c])
print('aux80k ' + ' '.join('%.4f+-%.3f' % (
    statistics.mean(res[c]), statistics.stdev(res[c])) for c in CP), flush=True)
print('DONE')
