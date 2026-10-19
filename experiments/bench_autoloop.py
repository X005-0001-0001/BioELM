"""自动循环判据探索（elastic 1.1.x 方向）：推理时按判据提前停，逐 batch 列自适应深度。

判据一（语义收敛）：连续子步的表示相对变化 ‖Δsem‖/‖sem‖ < ε → 停。
判据二（分布熵）：softmax 熵 < H_thr → 停。
用法（仓库根目录）：
    python experiments/bench_autoloop.py
环境变量 BIO_ELASTIC_SUBSET 可指定子集条数（CPU 建议 500）。
结论（见同分支 autoloop_notes.md）：语义判据在相同平均深度下与固定 K 平价（无免费午餐），
熵判据无判别信号（熵 ~4.3+，任何阈值都停不下来）。→ 保留存档，正式版自动循环需
"学习型停机"（局部规则可实现的 halting readout），列入路线图。
"""
import os
import sys
import time
import importlib.util

os.environ.setdefault('BIO_DEV', 'cpu')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import torch
torch.set_num_threads(min(16, os.cpu_count() or 8))

spec = importlib.util.spec_from_file_location('bench_el', os.path.join(BASE, 'experiments', 'bench_elastic.py'))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

NS, NV, TV, STEPS = B.NS, B.NV, B.TV, B.STEPS
tr_tok, tr_tgt, tr_msk = B.prep(B.tr_ids, B.tr_mask, NS, STEPS)
va_tok, va_tgt, va_msk = B.prep(B.va_ids, B.va_mask, NV, TV)

T0 = time.time()
net = B.ElasticNet(mode='consist', kmax=4, lam=1.0, h=192)
st = net.init_state(NS)
for s in range(STEPS):
    net.step(tr_tok[:, s], tr_tgt[:, s], st, train=True, mask=tr_msk[:, s])
print('trained %.1fs' % (time.time() - T0))

lay = net.layer


@torch.no_grad()
def eval_fixed(K):
    stv = net.init_state(NV)
    tot, cnt = torch.zeros(()), torch.zeros(())
    for t in range(TV):
        lg, _, _ = net.step(va_tok[:, t], va_tgt[:, t], stv, train=False, K=K)
        nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, va_tgt[:, t].view(-1, 1)).squeeze(1)
        tot = tot + (nll * va_msk[:, t]).sum()
        cnt = cnt + va_msk[:, t].sum()
    return float(tot / cnt.clamp(min=1.0))


@torch.no_grad()
def eval_auto(criterion, thr, kmax=6):
    stv = net.init_state(NV)
    tot, cnt = torch.zeros(()), torch.zeros(())
    depth_sum, depth_n = 0.0, 0
    for t in range(TV):
        x = net.E[va_tok[:, t]]
        prev_sem = None
        lg = sem = None
        used = kmax
        for k in range(1, kmax + 1):
            lg, sem = lay.cell(x, stv, trace=False)
            if criterion == 'sem':
                if prev_sem is not None:
                    rel = float(((sem - prev_sem).norm() / (prev_sem.norm() + 1e-9)))
                else:
                    rel = 1e9
                if rel < thr:
                    used = k
                    break
                prev_sem = sem.detach()
            else:
                ent = float(-(torch.softmax(lg, dim=-1) * torch.log_softmax(lg, dim=-1)).sum(-1).mean())
                if ent < thr:
                    used = k
                    break
        depth_sum += used * x.shape[0]
        depth_n += x.shape[0]
        nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, va_tgt[:, t].view(-1, 1)).squeeze(1)
        tot = tot + (nll * va_msk[:, t]).sum()
        cnt = cnt + va_msk[:, t].sum()
    return float(tot / cnt.clamp(min=1.0)), depth_sum / depth_n


print('fixed: ' + ' '.join('K%d=%.4f' % (k, eval_fixed(k)) for k in (1, 2, 3, 4)))
for thr in (0.5, 0.2, 0.1, 0.05, 0.02):
    ce, d = eval_auto('sem', thr)
    print('auto-sem thr=%.2f valCE=%.4f avgDepth=%.3f' % (thr, ce, d))
for thr in (0.35, 0.30, 0.25, 0.20, 0.15):
    ce, d = eval_auto('ent', thr)
    print('auto-ent thr=%.2f valCE=%.4f avgDepth=%.3f' % (thr, ce, d))
