"""时间条件注入在固定深训练中的检验（deep 场景）：K=4 单路线 CE，步数由 BIO_TC_STEPS 控制（默认 3000），h=192（干净 500 条子集）。
对照：plain（gb/db 冻结为零，不学习） vs timecond（每子步门控偏置 gb[k]/db[k] 用精确局部规则学习）。
注意：anytime 场景已证时间条件有害（expr-timecond）；本实验检验固定深场景是否反而有益。
"""
import os
import sys
import math
import time
import importlib.util

os.environ.setdefault('BIO_DEV', 'cpu')
os.environ.setdefault('BIO_ELASTIC_SUBSET', '500')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

import torch
torch.set_num_threads(8)

DEV = torch.device('cpu')
VOCAB = 2754
ETA = 2.0e-3
WD = 1.0e-4
BETA1, BETA2, EPS = 0.9, 0.99, 1e-6
KMAX = 4

spec = importlib.util.spec_from_file_location('bench_el', os.path.join(BASE, 'experiments', 'bench_elastic.py'))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

NS, NV, TV = B.NS, B.NV, B.TV
STEPS = int(os.environ.get('BIO_TC_STEPS', '3000'))
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
    def route(self, x, st, K, trace=True, collect=True):
        logits = sem = None
        hists = []
        for k in range(K):
            logits, sem, hist = self.cell(x, st, trace=trace, k=k)
            if trace and collect:
                hists.append(hist)
        return logits, sem, hists

    @torch.no_grad()
    def learn(self, sem, st, e=None, g_sem=None, hists=None):
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
            gb_g = torch.zeros(KMAX, self.h, device=DEV)
            db_g = torch.zeros(KMAX, self.h, device=DEV)
            for k, (ebg, ebd) in enumerate(hists):
                gb_g[k] = torch.einsum('bh,bh->h', Ls, ebg)
                db_g[k] = torch.einsum('bh,bh->h', Ls, ebd)
            grads += [('gb', gb_g), ('db', db_g)]
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
    def __init__(self, h=192, m=48, d=48, seed=43):
        gen = torch.Generator().manual_seed(seed)
        self.E = (torch.randn(VOCAB, d, generator=gen) * 0.6).to(DEV)
        self.layer = LayerT(d, h, m, gen)

    def init_state(self, batch):
        return self.layer.init_state(batch)

    @torch.no_grad()
    def step(self, tok_col, tgt_col, st, train=True, mask=None, learn_bias=True):
        lay = self.layer
        x = self.E[tok_col]
        if not train:
            logits, sem, _ = lay.route(x, st, KMAX, trace=False)
            return logits, None, None
        Bn = tok_col.numel()
        ar = torch.arange(Bn, device=DEV)
        logits, sem, hists = lay.route(x, st, KMAX, trace=True, collect=learn_bias)
        e = torch.softmax(logits, dim=-1)
        e[ar, tgt_col] = e[ar, tgt_col] - 1.0
        if mask is None:
            e = e / Bn
        else:
            e = e * mask.unsqueeze(1) / mask.sum().clamp(min=1.0)
        lay.learn(sem, st, e=e, hists=hists if learn_bias else None)
        return logits, None, None


def run(label, learn_bias):
    net = NetT()
    st = net.init_state(NS)
    t0 = time.time()
    for s in range(STEPS):
        net.step(tr_tok[:, s], tr_tgt[:, s], st, train=True, mask=tr_msk[:, s], learn_bias=learn_bias)
    dt = time.time() - t0
    stv = net.init_state(NV)
    tot, cnt = torch.zeros(()), torch.zeros(())
    for t in range(TV):
        lg, _, _ = net.step(va_tok[:, t], va_tgt[:, t], stv, train=False)
        nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, va_tgt[:, t].view(-1, 1)).squeeze(1)
        tot = tot + (nll * va_msk[:, t]).sum()
        cnt = cnt + va_msk[:, t].sum()
    ce = float(tot / cnt.clamp(min=1.0))
    print('[%s] %.1fs valCE=%.4f' % (label, dt, ce), flush=True)


run('fixedK4-plain', False)
run('fixedK4-timecond', True)
print('DONE')
