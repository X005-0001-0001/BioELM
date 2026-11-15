"""Tiny smoke for new lab switches: baseline vs switch, 100 steps, 16 streams."""
import os
os.environ['BIO_DEV'] = 'cpu'
import sys
import importlib.util

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

spec = importlib.util.spec_from_file_location('lab', os.path.join(BASE, 'experiments', 'struct_lab.py'))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
L.torch.set_num_threads(2)

BASE_CFG = {'K': 1, 'TAU': 1.0, 'h': 128, 'd': 96, 'm': 96, 'streams': 16,
            'steps': 100, 'checkpoints': (100,), 'EMA': 0.999, 'LSM': 0.05, 'SRMS': 1}

sw = sys.argv[1] if len(sys.argv) > 1 else 'COSHEAD'
base = L.run(dict(BASE_CFG, seed=43))
print('base %.6f' % base[100], flush=True)
sw_out = L.run(dict(BASE_CFG, seed=43, **{sw: 1}))
print('%s %.6f' % (sw, sw_out[100]), flush=True)
print('DONE')
