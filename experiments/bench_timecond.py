"""子步时间条件注入（t-conditioning）探索（负结果留档）。

每个子步 k 的门控加可学习偏置 gb[k]/db[k]（零初始化 → 起点与基线等价）。
梯度用与 b_g/b_d 完全相同的资格迹在"该子步时刻"的取值——精确在线 RTRL，无回传。
用法（仓库根目录）：
    python experiments/bench_timecond.py
环境变量 BIO_ELASTIC_SUBSET 可指定子集条数（CPU 建议 500）。
结论（expr-timecond 分支 notes）：深预算略好、浅预算与预算外推明显变差 → 不进入产品。
"""
import os
import sys
import math
import time

os.environ.setdefault('BIO_DEV', 'cpu')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import torch
torch.set_num_threads(min(16, os.cpu_count() or 8))

DEV = torch.device(os.environ.get('BIO_DEV', 'cpu'))
VOCAB = 2754
ETA = 2.0e-3
WD = 1.0e-4
BETA1, BETA2, EPS = 0.9, 0.99, 1e-6
KMAX = 4


import importlib.util
spec = importlib.util.spec_from_file_location('bench_el', os.path.join(BASE, 'experiments', 'bench_elastic.py'))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

NS, NV, TV, STEPS = B.NS, B.NV, B.TV, B.STEPS
tr_tok, tr_tgt, tr_msk = B.prep(B.tr_ids, B.tr_mask, NS, STEPS)
va_tok, va_tgt, va_msk = B.prep(B.va_ids, B.va_mask, NV, TV)


class LayerT:
    def __init__(self, d_in, h, m, gen):
        self.d_in, self.h, self.m = d_in, h, m
        self.W_g = ((torch.rand(h, d_in, generator=gen) * 2 - 1) / math.sqrt(d_in)).to(DEV)
        self.b_g = torch.zeros(h).to(DEV)
        self.W_d = ((torch.rand(h, d_in, generator=gen) * 2 - 1) / math.sqrt(d_in)).to(DEV)
        tau0 = math.sqrt(h)
        d0 = 1.0 - math.log(2.0) / tau0
        self.b_d = torch.full((h,), math.log(d0 / (1.0 - d0))).to(DEV)
        self.B = (torch.randn(h, d_in, generator=gen) * (1.0 / math.sqrt(d_in))).to(DEV)
        self.Wp = ((torch.rand(m, h, generator=gen) * 2 - 1) / math.sqrt(h)).to(DEV)
        self.bp = torch.zeros(m).to(DEV)
        self.head = ((torch.rand(VOCAB, m, generator=gen) * 2 - 1) / math.sqrt(m)).to(DEV)
        self.bout = torch.zeros(VOCAB).to(DEV)
        self.gb = torch.zeros(KMAX, h).to(DEV)
        self.db = torch.zeros(KMAX, h).to(DEV)
        self.t = 0
        self._meta = {k: (torch.zeros_like(p), torch.zeros_like(p))
                      for k, p in self.params().items()}

    def params(self):
        return {'W_g': self.W_g, 'b_g': self.b_g, 'W_d': self.W_d, 'b_d': self.b_d,
                'B': self.B, 'Wp': self.Wp, 'bp': self.bp,
                'head': self.head, 'bout': self.bout,
                'gb': self.gb, 'db': self.db}

    def init_state(self, batch):
        return {
            's': torch.zeros(batch, self.h, device=DEV),
            'eK': torch.zeros(3, batch, self.h, self.d_in, device=DEV),
            'e_bg': torch.zeros(batch, self.h, device=DEV),
            'e_bd': torch.zeros(batch, self.h, device=DEV),
        }

    def clone_state(self, st):
        return {k: v.clone() for k, v in st.items()}

    def copy_state_(self, dst, src):
        for k in dst:
            dst[k].copy_(src[k])

    @torch.no_grad()
    def cell(self, x, st, trace=True, k=0):
        g = torch.sigmoid(x @ self.W_g.t() + self.b_g + self.gb[k % KMAX])
        d = torch.sigmoid(x @ self.W_d.t() + self.b_d + self.db[k % KMAX])
        u_raw = x @ self.B.t()
        u = g * u_raw
        s_buf = st['s']
        if trace:
            c_g = (g * (1.0 - g)) * u_raw
            c_d = (d * (1.0 - d)) * s_buf
        s_buf.mul_(d).add_(u)
        sem = torch.sigmoid(s_buf @ self.Wp.t() + self.bp)
        logits = sem @ self.head.t() + self.bout
        hist = None
        if trace:
            eK = st['eK']
            eK.mul_(d.unsqueeze(-1).unsqueeze(0))
            eK.addcmul_(torch.stack((c_g, c_d, g), dim=0).unsqueeze(-1),
                        x.unsqueeze(0).unsqueeze(2))
            st['e_bg'].mul_(d).add_(c_g)
            st['e_bd'].mul_(d).add_(c_d)
            hist = (st['e_bg'].clone(), st['e_bd'].clone())
        return logits, sem, hist

    @torch.no_grad()
    def route(self, x, st, K, trace=True):
        logits = sem = None
        hists = []
        for k in range(K):
            logits, sem, hist = self.cell(x, st, trace=trace, k=k)
            if hist is not None:
                hists.append(hist)
        return logits, sem, hists

    @torch.no_grad()
    def learn(self, sem, st, e=None, g_sem=None, hists=None, K=0):
        s = st['s']
        if g_sem is None:
            g_sem = (e @ self.head) * (sem * (1.0 - sem))
        Ls = g_sem @ self.Wp
        eK = st['eK']
        grads = [
            ('W_g', torch.einsum('bh,bhd->hd', Ls, eK[0])),
            ('b_g', torch.einsum('bh,bh->h', Ls, st['e_bg'])),
            ('W_d', torch.einsum('bh,bhd->hd', Ls, eK[1])),
            ('b_d', torch.einsum('bh,bh->h', Ls, st['e_bd'])),
            ('B', torch.einsum('bh,bhd->hd', Ls, eK[2])),
            ('Wp', g_sem.t() @ s),
            ('bp', g_sem.sum(0)),
            ('head', e.t() @ sem),
            ('bout', e.sum(0)),
        ]
        if hists:
            gb_g = torch.zeros(KMAX, self.h)
            db_g = torch.zeros(KMAX, self.h)
            for k, (ebg, ebd) in enumerate(hists):
                gb_g[k] = torch.einsum('bh,bh->h', Ls, ebg)
                db_g[k] = torch.einsum('bh,bh->h', Ls, ebd)
            grads += [('gb', gb_g.to(DEV)), ('db', db_g.to(DEV))]
        self.apply(grads)

    @torch.no_grad()
    def apply(self, grads):
        self.t += 1
        b1c = 1.0 - BETA1 ** self.t
        b2c = 1.0 - BETA2 ** self.t
        names = [n for n, _ in grads]
        gs = [g for _, g in grads]
        params = [self.params()[n] for n in names]
        mm = [self._meta[n][0] for n in names]
        vv = [self._meta[n][1] for n in names]
        torch._foreach_mul_(mm, BETA1)
        torch._foreach_add_(mm, gs, alpha=1.0 - BETA1)
        torch._foreach_mul_(vv, BETA2)
        torch._foreach_addcmul_(vv, gs, gs, value=1.0 - BETA2)
        mhat = torch._foreach_mul(mm, 1.0 / b1c)
        vhat = torch._foreach_sqrt(torch._foreach_div(vv, b2c))
        torch._foreach_add_(vhat, EPS)
        upd = torch._foreach_div(mhat, vhat)
        p2 = [p for p in params if p.dim() == 2]
        if p2:
            torch._foreach_mul_(p2, 1.0 - WD)
        torch._foreach_add_(params, upd, alpha=-ETA)


class NetT:
    def __init__(self, lam=1.0, h=192, m=48, d=48, seed=43, rng_seed=1234):
        import random
        self.lam = lam
        self.rng = random.Random(rng_seed)
        gen = torch.Generator().manual_seed(seed)
        self.E = (torch.randn(VOCAB, d, generator=gen) * 0.6).to(DEV)
        self.layer = LayerT(d, h, m, gen)

    def init_state(self, batch):
        return self.layer.init_state(batch)

    @torch.no_grad()
    def step(self, tok_col, tgt_col, st, train=True, mask=None, K=None):
        lay = self.layer
        x = self.E[tok_col]
        if not train:
            k = KMAX if K is None else K
            logits, sem, _ = lay.route(x, st, k, trace=False)
            return logits, None, None
        Bn = tok_col.numel()
        ar = torch.arange(Bn, device=DEV)
        st_f = lay.clone_state(st)
        st_s = lay.clone_state(st)
        logits, sem_f, hists_f = lay.route(x, st_f, KMAX, trace=True)
        m_short = self.rng.randint(1, KMAX - 1)
        logits_s, sem_s, hists_s = lay.route(x, st_s, m_short, trace=True)
        e_f = torch.softmax(logits, dim=-1)
        e_f[ar, tgt_col] = e_f[ar, tgt_col] - 1.0
        e_s = torch.softmax(logits_s, dim=-1)
        e_s[ar, tgt_col] = e_s[ar, tgt_col] - 1.0
        if mask is None:
            e_f = e_f / Bn
            e_s = e_s / Bn
            sc = None
        else:
            sc = (mask.unsqueeze(1) / mask.sum().clamp(min=1.0))
            e_f = e_f * sc
            e_s = e_s * sc
        g_ce = (e_s @ lay.head) * (sem_s * (1.0 - sem_s))
        g_cons = self.lam * (sem_s - sem_f.detach()) * (sem_s * (1.0 - sem_s))
        if sc is not None:
            g_cons = g_cons * sc
        lay.learn(sem_s, st_s, e=e_s, g_sem=g_ce + g_cons, hists=hists_s, K=m_short)
        lay.learn(sem_f, st_f, e=e_f, hists=hists_f, K=KMAX)
        lay.copy_state_(st, st_f)
        return logits, None, None


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


net = NetT()
st = net.init_state(NS)
t0 = time.time()
for s in range(STEPS):
    net.step(tr_tok[:, s], tr_tgt[:, s], st, train=True, mask=tr_msk[:, s])
dt = time.time() - t0
sc = evaluate(net)
print('[timecond-kmax4] %.1fs | %s' % (dt, ' '.join('K%d=%.4f' % (i + 1, c) for i, c in enumerate(sc))))
