"""K 调度（前慢后深）对照：一致性训练中 KMAX 分段递增 vs 固定（负结果留档）。

用法（仓库根目录）：
    python experiments/bench_ksched.py
环境变量 BIO_ELASTIC_SUBSET 可指定子集条数（CPU 建议 500）。
结论（expr-ksched 分支 notes）：所有调度（2→4、2→4→6、1→2→4）均劣于固定 KMAX=4，
浅预算尤其受损 → 产品维持固定 KMAX=4。
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


def evaluate(net):
    scores = []
    for k in range(1, 7):
        stv = net.init_state(NV)
        tot, cnt = torch.zeros(()), torch.zeros(())
        for t in range(TV):
            lg, _, _ = net.step(va_tok[:, t], va_tgt[:, t], stv, train=False, K=k)
            nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, va_tgt[:, t].view(-1, 1)).squeeze(1)
            tot = tot + (nll * va_msk[:, t]).sum()
            cnt = cnt + va_msk[:, t].sum()
        scores.append(float(tot / cnt.clamp(min=1.0)))
    return scores


def run(label, segments):
    net = B.ElasticNet(mode='consist', kmax=max(km for km, _ in segments), lam=1.0, h=192)
    net.kmax = segments[0][0]
    st = net.init_state(NS)
    segs = list(segments)
    t0 = time.time()
    for s in range(STEPS):
        while len(segs) > 1 and s >= STEPS * segs[0][1]:
            segs.pop(0)
        net.kmax = segs[0][0]
        net.step(tr_tok[:, s], tr_tgt[:, s], st, train=True, mask=tr_msk[:, s])
    dt = time.time() - t0
    sc = evaluate(net)
    print('[%s] %.1fs | %s' % (label, dt, ' '.join('K%d=%.4f' % (i + 1, c) for i, c in enumerate(sc))))
    return sc


run('kmax4-fixed', [(4, 1.01)])
run('sched-2-4', [(2, 0.333), (4, 1.01)])
run('sched-2-4-6', [(2, 0.333), (4, 0.666), (6, 1.01)])
run('sched-1-2-4', [(1, 0.25), (2, 0.55), (4, 1.01)])
