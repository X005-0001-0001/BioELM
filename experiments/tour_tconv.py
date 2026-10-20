"""tour-tour-tconv runner：输入时间因果卷积（TCONV=3）
用 struct_lab（main）的开关排；本分支只固定该位置的配置。"""
import os
import sys
import importlib.util

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

spec = importlib.util.spec_from_file_location('lab', os.path.join(BASE, 'experiments', 'struct_lab.py'))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

SEEDS = (43, 44, 45)
for cfg in [dict(K=2, TCONV=3), dict(K=4, TCONV=3)]:
    for seed in SEEDS:
        L.run(dict(cfg, seed=seed, steps=20000, checkpoints=(5000, 20000)))
