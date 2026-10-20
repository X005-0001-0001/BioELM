"""学习型停机 v2：平滑边际目标 + 逐 token 阈值（CPU, h=192, consist KMAX=4, 3000 步）。
复用仓库 bench_halting 的 Layer/Net（数据、模型、训练目标一致），仅改：
1) 停机目标用相邻步平滑后的 CE：ce_s[k]=½(ce[k]+ce[k+1])，y_k=1[ce_s[k] ≤ ce_s[k+1]]
2) 推理时逐 token（逐行）阈值 p_halt>0.5 即停，报告平均深度与深度分布
对照：固定 K2/K3；v1（batch 均值停机）。
"""
import os
import sys
import importlib.util

os.environ.setdefault('BIO_DEV', 'cpu')
os.environ.setdefault('BIO_ELASTIC_SUBSET', '500')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import torch
torch.set_num_threads(8)

spec = importlib.util.spec_from_file_location('bh', os.path.join(BASE, 'experiments', 'bench_halting.py'))
BH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BH)

KMAX = BH.KMAX
NS, NV, TV, STEPS = BH.NS, BH.NV, BH.TV, BH.STEPS
tr_tok, tr_tgt, tr_msk = BH.tr_tok, BH.tr_tgt, BH.tr_msk
va_tok, va_tgt, va_msk = BH.va_tok, BH.va_tgt, BH.va_msk


def train_halter_v2(self, lg_list, sem_list, tgt, mask):
    ce = []
    for lg in lg_list:
        nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, tgt.view(-1, 1)).squeeze(1)
        ce.append(nll.detach())
    K = len(ce)
    ce_s = []
    for k in range(K):
        if k + 1 < K:
            ce_s.append(0.5 * (ce[k] + ce[k + 1]))
        else:
            ce_s.append(ce[k])
    pairs_sem, pairs_y = [], []
    for k in range(K - 1):
        y = (ce_s[k] <= ce_s[k + 1]).float()
        pairs_sem.append(sem_list[k])
        pairs_y.append(y)
    pairs_sem.append(sem_list[K - 1])
    pairs_y.append(torch.ones_like(ce[0]))
    sem_all = torch.cat(pairs_sem, 0)
    y_all = torch.cat(pairs_y, 0)
    if mask is not None:
        m_all = torch.cat([mask] * K, 0)
    else:
        m_all = torch.ones_like(y_all)
    p = torch.sigmoid(sem_all @ self.hw + self.hb)
    wgt = m_all / m_all.sum().clamp(min=1.0)
    err = (p - y_all) * wgt
    self.hw -= BH.ETA_H * (err @ sem_all)
    self.hb -= BH.ETA_H * err.sum()


BH.Net.train_halter = train_halter_v2


@torch.no_grad()
def eval_halt_pertoken(net):
    stv = net.init_state(NV)
    tot, cnt = torch.zeros(()), torch.zeros(())
    depth_sum, depth_n = 0.0, 0
    dist = [0.0] * (KMAX + 1)
    for t in range(TV):
        x = net.E[va_tok[:, t]]
        lay = net.layer
        B = x.shape[0]
        halted = torch.zeros(B, dtype=torch.bool)
        depth = torch.full((B,), KMAX, dtype=torch.float32)
        final_lg = None
        lg = sem = None
        for k in range(KMAX):
            lg, sem = lay.cell(x, stv, trace=False)
            if final_lg is None:
                final_lg = lg.clone()
            p = torch.sigmoid(sem @ net.hw + net.hb)
            newly = (~halted) & (p > 0.5) & (k >= 1)
            if newly.any():
                final_lg[newly] = lg[newly]
                depth[newly] = float(k + 1)
                halted = halted | newly
            if halted.all():
                break
        final_lg[~halted] = lg[~halted]
        m = va_msk[:, t]
        nll = torch.logsumexp(final_lg, dim=-1) - final_lg.gather(1, va_tgt[:, t].view(-1, 1)).squeeze(1)
        tot = tot + (nll * m).sum()
        cnt = cnt + m.sum()
        depth_sum += float((depth * m).sum())
        depth_n += float(m.sum())
        for kk in range(1, KMAX + 1):
            dist[kk] += float(((depth == kk).float() * m).sum())
    return float(tot / cnt.clamp(min=1.0)), depth_sum / max(depth_n, 1.0), [d / max(depth_n, 1.0) for d in dist]


@torch.no_grad()
def eval_fixed(net, K):
    stv = net.init_state(NV)
    tot, cnt = torch.zeros(()), torch.zeros(())
    for t in range(TV):
        lg, _, _ = net.step(va_tok[:, t], va_tgt[:, t], stv, train=False, K_eval=K)
        nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, va_tgt[:, t].view(-1, 1)).squeeze(1)
        tot = tot + (nll * va_msk[:, t]).sum()
        cnt = cnt + va_msk[:, t].sum()
    return float(tot / cnt.clamp(min=1.0))


net = BH.Net()
st = net.init_state(NS)
for s in range(STEPS):
    net.step(tr_tok[:, s], tr_tgt[:, s], st, train=True, mask=tr_msk[:, s])

print('fixed: ' + ' '.join('K%d=%.4f' % (k, eval_fixed(net, k)) for k in (1, 2, 3, 4)), flush=True)
ce, d, dist = eval_halt_pertoken(net)
print('halt-v2 per-token: valCE=%.4f avgDepth=%.3f dist=%s'
      % (ce, d, ' '.join('D%d=%.3f' % (i, x) for i, x in enumerate(dist) if i > 0)), flush=True)
print('DONE')
