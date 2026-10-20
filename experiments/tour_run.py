"""淘汰赛第二轮：深度 × 3 种子 × 2 检查点（GPU，20k 步，全语料）。"""
import os
import sys
import statistics
import importlib.util

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

spec = importlib.util.spec_from_file_location('lab', os.path.join(BASE, 'experiments', 'struct_lab.py'))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

CONFIGS = [('K2', 2), ('K3', 3), ('K4', 4), ('K6', 6)]
SEEDS = (43, 44, 45)
STEPS = 20000
CP = (5000, 20000)

res = {}
for name, K in CONFIGS:
    res[name] = {c: [] for c in CP}
    for seed in SEEDS:
        out = L.run({'K': K, 'MIX': 'none', 'TCONV': 0, 'seed': seed,
                     'steps': STEPS, 'checkpoints': CP})
        for c in CP:
            res[name][c].append(out[c])

print('\n=== ROUND 2 (mean ± std over %d seeds, valCE) ===' % len(SEEDS))
hdr = 'conf   ' + ' '.join('@%-9d' % c for c in CP)
print(hdr)
for name, _ in CONFIGS:
    cells = []
    for c in CP:
        v = res[name][c]
        cells.append('%.4f±%.3f' % (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0))
    print('%-6s ' % name + ' '.join(cells))
print('DONE')
