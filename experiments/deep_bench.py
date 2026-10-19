"""deep 分支探索台：[D1] 循环深度 vs 层堆叠 vs 反馈对齐（全零 autograd）。

用法（仓库根目录执行）：
    python experiments/deep_bench.py r1            # 单层 / 循环K2 / 层监督L2 / FA L2 / 层监督L3
    python experiments/deep_bench.py r2            # 循环 K3/K4、读出反馈环、h=128 对照
    python experiments/deep_bench.py r3            # 6000 步缩放对照
环境变量 BIO_DEEP_DATA 可指定语料（默认 dsr1_v5_0001.jsonl；CPU 建议 500 条子集）。
零 autograd：本文件不出现 loss.backward()，全部更新为局部规则。
"""
import os
import sys
import time
import math
import importlib.util
import tempfile

os.environ.setdefault('BIO_DEV', 'cpu')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

import torch
torch.set_num_threads(min(16, os.cpu_count() or 8))

DEV = torch.device(os.environ.get('BIO_DEV', 'cpu'))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = load(os.path.join(BASE, 'bioelm_elite_1.1.4.py'), 'b114')
P.LOG_FILE = os.path.join(tempfile.gettempdir(), 'deep_bench_log.txt')
P.init_log_file()
tok, _ = P._load_any_tokenizer()

DATA = os.environ.get('BIO_DEEP_DATA', os.path.join(BASE, 'dsr1_v5_0001.jsonl'))
SUBSET = int(os.environ.get('BIO_DEEP_SUBSET', '0'))
if SUBSET > 0:
    sub = os.path.join(tempfile.gettempdir(), 'deep_bench_sub.jsonl')
    with open(DATA, encoding='utf-8') as f, open(sub, 'w', encoding='utf-8') as g:
        for i, line in enumerate(f):
            if i >= SUBSET:
                break
            g.write(line)
    DATA = sub

tr_ids, tr_mask, va_ids, va_mask = P.build_corpus(tok, DATA)

NS = 32
NV = 8
TV = 200
STEPS = 3000
VOCAB = tok.vocab_size
ETA = 2.0e-3
WD = 1.0e-4
BETA1, BETA2, EPS = 0.9, 0.99, 1e-6


def prep(ids, mask, ns, T):
    need = ns * T + 1
    tv = torch.tensor(ids[:need - 1], dtype=torch.long).view(ns, T)
    tg = torch.tensor(ids[1:need], dtype=torch.long).view(ns, T)
    mk = torch.tensor(mask[1:need], dtype=torch.float32).view(ns, T)
    return tv, tg, mk


class Layer:
    def __init__(self, d_in, h, m, gen, with_head):
        self.d_in, self.h, self.m = d_in, h, m
        self.with_head = with_head
        self.W_g = ((torch.rand(h, d_in, generator=gen) * 2 - 1) / math.sqrt(d_in)).to(DEV)
        self.b_g = torch.zeros(h).to(DEV)
        self.W_d = ((torch.rand(h, d_in, generator=gen) * 2 - 1) / math.sqrt(d_in)).to(DEV)
        tau0 = math.sqrt(h)
        d0 = 1.0 - math.log(2.0) / tau0
        self.b_d = torch.full((h,), math.log(d0 / (1.0 - d0))).to(DEV)
        self.B = (torch.randn(h, d_in, generator=gen) * (1.0 / math.sqrt(d_in))).to(DEV)
        self.Wp = ((torch.rand(m, h, generator=gen) * 2 - 1) / math.sqrt(h)).to(DEV)
        self.bp = torch.zeros(m).to(DEV)
        if with_head:
            self.head = ((torch.rand(VOCAB, m, generator=gen) * 2 - 1) / math.sqrt(m)).to(DEV)
            self.bout = torch.zeros(VOCAB).to(DEV)
        else:
            self.head = None
            self.bout = None
        self.t = 0
        self._meta = {k: (torch.zeros_like(p), torch.zeros_like(p))
                      for k, p in self.params().items()}

    def params(self):
        d = {'W_g': self.W_g, 'b_g': self.b_g, 'W_d': self.W_d, 'b_d': self.b_d,
             'B': self.B, 'Wp': self.Wp, 'bp': self.bp}
        if self.head is not None:
            d['head'] = self.head
            d['bout'] = self.bout
        return d

    def init_state(self, batch):
        return {
            's': torch.zeros(batch, self.h, device=DEV),
            'eK': torch.zeros(3, batch, self.h, self.d_in, device=DEV),
            'e_bg': torch.zeros(batch, self.h, device=DEV),
            'e_bd': torch.zeros(batch, self.h, device=DEV),
        }

    @torch.no_grad()
    def cell(self, x, st, trace=True):
        g = torch.sigmoid(x @ self.W_g.t() + self.b_g)
        d = torch.sigmoid(x @ self.W_d.t() + self.b_d)
        u_raw = x @ self.B.t()
        u = g * u_raw
        s_buf = st['s']
        if trace:
            c_g = (g * (1.0 - g)) * u_raw
            c_d = (d * (1.0 - d)) * s_buf
        s_buf.mul_(d).add_(u)
        sem = torch.sigmoid(s_buf @ self.Wp.t() + self.bp)
        logits = None
        if self.head is not None:
            logits = sem @ self.head.t() + self.bout
        if trace:
            eK = st['eK']
            eK.mul_(d.unsqueeze(-1).unsqueeze(0))
            eK.addcmul_(torch.stack((c_g, c_d, g), dim=0).unsqueeze(-1),
                        x.unsqueeze(0).unsqueeze(2))
            st['e_bg'].mul_(d).add_(c_g)
            st['e_bd'].mul_(d).add_(c_d)
        return logits, sem

    @torch.no_grad()
    def learn(self, sem, st, e=None, g_sem=None):
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
        ]
        if self.head is not None:
            grads += [('head', e.t() @ sem), ('bout', e.sum(0))]
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


class Net:
    """mode: 'single' | 'loop' | 'loopfb' | 'stacked' | 'fa'"""

    def __init__(self, mode, L=1, K=1, h=192, m=48, d=48, seed=43):
        self.mode, self.L, self.K = mode, L, K
        gen = torch.Generator().manual_seed(seed)
        self.E = (torch.randn(VOCAB, d, generator=gen) * 0.6).to(DEV)
        layers = []
        for i in range(L):
            d_in = d if i == 0 else m
            with_head = (mode == 'stacked') or (i == L - 1)
            layers.append(Layer(d_in, h, m, gen, with_head))
        self.layers = layers
        self.F = None
        if mode == 'fa' and L > 1:
            Fs = []
            for i in range(L - 1):
                Fs.append(((torch.rand(VOCAB, m, generator=gen) * 2 - 1) / math.sqrt(m)).to(DEV))
            self.F = Fs

    def n_params(self):
        n = self.E.numel()
        for lay in self.layers:
            n += sum(p.numel() for p in lay.params().values())
        return n

    def init_state(self, batch):
        return [lay.init_state(batch) for lay in self.layers]

    @torch.no_grad()
    def step(self, tok_col, tgt_col, st, train=True, mask=None):
        x = self.E[tok_col]
        logits_list = []
        sems = []
        for i, lay in enumerate(self.layers):
            if self.mode == 'loop':
                for _ in range(self.K):
                    logits, sem = lay.cell(x, st[i], trace=train)
            elif self.mode == 'loopfb':
                logits, sem = lay.cell(x, st[i], trace=train)
                for _ in range(self.K - 1):
                    logits, sem = lay.cell(sem, st[i], trace=train)
            else:
                logits, sem = lay.cell(x, st[i], trace=train)
            logits_list.append(logits)
            sems.append(sem)
            x = sem
        if not train:
            return logits_list[-1], None, None
        B = tok_col.numel()
        ar = torch.arange(B, device=DEV)
        if self.mode == 'stacked':
            for i, lay in enumerate(self.layers):
                lg = logits_list[i]
                p = torch.softmax(lg, dim=-1)
                e = p
                e[ar, tgt_col] = e[ar, tgt_col] - 1.0
                if mask is None:
                    e = e / B
                else:
                    mm = mask.unsqueeze(1)
                    e = e * mm / mm.sum().clamp(min=1.0)
                lay.learn(sems[i], st[i], e=e)
        else:
            lg = logits_list[-1]
            p = torch.softmax(lg, dim=-1)
            e = p
            e[ar, tgt_col] = e[ar, tgt_col] - 1.0
            if self.mode == 'fa' and self.L > 1:
                for i in range(self.L - 1):
                    g_sem = (e @ self.F[i]) * (sems[i] * (1.0 - sems[i]))
                    scale = (1.0 / B) if mask is None else (mask.unsqueeze(1) / mask.sum().clamp(min=1.0))
                    self.layers[i].learn(sems[i], st[i], g_sem=g_sem * scale)
                ee = e / B if mask is None else e * mask.unsqueeze(1) / mask.sum().clamp(min=1.0)
                self.layers[-1].learn(sems[-1], st[-1], e=ee)
            else:
                if mask is None:
                    ee = e / B
                else:
                    ee = e * mask.unsqueeze(1) / mask.sum().clamp(min=1.0)
                self.layers[-1].learn(sems[-1], st[-1], e=ee)
        nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, tgt_col.view(-1, 1)).squeeze(1)
        if mask is None:
            nll_sum = nll.sum()
            cnt = torch.tensor(float(B), device=DEV)
        else:
            nll_sum = (nll * mask).sum()
            cnt = mask.sum()
        return lg, nll_sum, cnt


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'r1'
    va_tok, va_tgt, va_msk = prep(va_ids, va_mask, NV, TV)

    def run(label, steps=STEPS, **kw):
        steps = int(steps)
        tr_tok, tr_tgt, tr_msk = prep(tr_ids, tr_mask, NS, steps)
        net = Net(**kw)
        st = net.init_state(NS)
        t0 = time.time()
        for s in range(steps):
            net.step(tr_tok[:, s], tr_tgt[:, s], st, train=True, mask=tr_msk[:, s])
        dt = time.time() - t0
        stv = net.init_state(NV)
        tot, cnt = torch.zeros(()), torch.zeros(())
        for t in range(TV):
            lg, _, _ = net.step(va_tok[:, t], va_tgt[:, t], stv, train=False)
            nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, va_tgt[:, t].view(-1, 1)).squeeze(1)
            tot = tot + (nll * va_msk[:, t]).sum()
            cnt = cnt + va_msk[:, t].sum()
        ce = float(tot / cnt.clamp(min=1.0))
        print('[%s] mode=%s L=%d K=%d h=%d params=%d %.1fs (%.1f st/s) valCE=%.4f'
              % (label, kw.get('mode'), kw.get('L', 1), kw.get('K', 1), kw.get('h', 192),
                 net.n_params(), dt, steps / dt, ce))
        return ce

    if which == 'r1':
        run('single-192', mode='single', h=192)
        run('loop2-192', mode='loop', K=2, h=192)
        run('stacked2-192', mode='stacked', L=2, h=192)
        run('fa2-192', mode='fa', L=2, h=192)
        run('stacked3-192', mode='stacked', L=3, h=192)
    if which == 'r2':
        run('loop3-192', mode='loop', K=3, h=192)
        run('loop4-192', mode='loop', K=4, h=192)
        run('loopfb2-192', mode='loopfb', K=2, h=192)
        run('loopfb3-192', mode='loopfb', K=3, h=192)
        run('loop2-128', mode='loop', K=2, h=128)
        run('single-128', mode='single', h=128)
    if which == 'r3':
        run('single-192-6000', mode='single', h=192, steps=6000)
        run('loop2-192-6000', mode='loop', K=2, h=192, steps=6000)


if __name__ == '__main__':
    main()
