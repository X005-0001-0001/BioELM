"""Round 11 (CPU): readout width M sweep. 128 streams x 5k x 3 seeds.
Baseline M=48; candidates M=96, M=192. Same h=192/d=48."""
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

SEEDS = (43, 44, 45)
CP = (2500, 5000)
CFGS = [('M48', 48), ('M96', 96), ('M192', 192)]
rows = []
for name, m in CFGS:
    res = {c: [] for c in CP}
    for seed in SEEDS:
        out = L.run({'K': 2, 'TAU': 1.0, 'm': m, 'seed': seed,
                     'streams': 128, 'steps': 5000, 'checkpoints': CP})
        for c in CP:
            res[c].append(out[c])
    line = ('%-6s ' % name + ' '.join('%.4f+-%.3f' % (
        statistics.mean(res[c]), statistics.stdev(res[c])) for c in CP))
    print(line, flush=True)
    rows.append(line)

print('\n=== ROUND 11 READOUT WIDTH (128 streams x 5k, mean +- std over 3 seeds) ===')
for r in rows:
    print(r)
print('DONE')
