"""elastic 分支探索台：[E1] 弹性深度（LoopFormer 式变长轨迹 + 捷径一致性，全零 autograd）。

用法（仓库根目录执行）：
    python experiments/bench_elastic.py r1     # 头条表：固定K4 / 随机K / 一致性 λ=1.0 的 anytime@K=1..6
    python experiments/bench_elastic.py r2     # λ 消融（0.3 / 3.0）+ 6000 步缩放
环境变量 BIO_ELASTIC_DATA 可指定语料（默认 dsr1_v5_0001.jsonl；CPU 建议 500 条子集）。
零 autograd：本文件不出现 loss.backward()；一致性目标是逐张量 detached（stop-grad 天然成立）。
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


P = load(os.path.join(BASE, 'bioelm_elite_1.1.5.py'), 'b115')
P.LOG_FILE = os.path.join(tempfile.gettempdir(), 'elastic_bench_log.txt')
P.init_log_file()
tok, _ = P._load_any_tokenizer()

DATA = os.environ.get('BIO_ELASTIC_DATA', os.path.join(BASE, 'dsr1_v5_0001.jsonl'))
SUBSET = int(os.environ.get('BIO_ELASTIC_SUBSET', '0'))
if SUBSET > 0:
    sub = os.path.join(tempfile.gettempdir(), 'elastic_bench_sub.jsonl')
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
        self.t = 0
        self._meta = {k: (torch.zeros_like(p), torch.zeros_like(p))
                      for k, p in self.params().items()}

    def params(self):
        return {'W_g': self.W_g, 'b_g': self.b_g, 'W_d': self.W_d, 'b_d': self.b_d,
                'B': self.B, 'Wp': self.Wp, 'bp': self.bp,
                'head': self.head, 'bout': self.bout}

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
        logits = sem @ self.head.t() + self.bout
        if trace:
            eK = st['eK']
            eK.mul_(d.unsqueeze(-1).unsqueeze(0))
            eK.addcmul_(torch.stack((c_g, c_d, g), dim=0).unsqueeze(-1),
                        x.unsqueeze(0).unsqueeze(2))
            st['e_bg'].mul_(d).add_(c_g)
            st['e_bd'].mul_(d).add_(c_d)
        return logits, sem

    def route(self, tok, st, K, trace=True):
        logits = sem = None
        for _ in range(K):
            logits, sem = self.cell(tok, st, trace=trace)
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
            ('head', e.t() @ sem),
            ('bout', e.sum(0)),
        ]
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


class ElasticNet:
    """mode: 'fixed'（固定 K_MAX，朴素基线）| 'random'（随机深度仅 CE）| 'consist'（完整+捷径一致性）"""

    def __init__(self, mode='consist', kmax=4, lam=1.0, h=192, m=48, d=48, seed=43, rng_seed=1234):
        self.mode, self.kmax, self.lam = mode, kmax, lam
        self.rng = __import__('random').Random(rng_seed)
        gen = torch.Generator().manual_seed(seed)
        self.E = (torch.randn(VOCAB, d, generator=gen) * 0.6).to(DEV)
        self.layer = Layer(d, h, m, gen)

    def n_params(self):
        return self.E.numel() + sum(p.numel() for p in self.layer.params().values())

    def init_state(self, batch):
        return self.layer.init_state(batch)

    @torch.no_grad()
    def step(self, tok_col, tgt_col, st, train=True, mask=None, K=None):
        lay = self.layer
        x = self.E[tok_col]
        if not train:
            k = self.kmax if K is None else K
            logits, sem = lay.route(x, st, k, trace=False)
            return logits, None, None
        B = tok_col.numel()
        ar = torch.arange(B, device=DEV)
        if self.mode == 'consist' and self.kmax > 1:
            st_f = lay.clone_state(st)
            st_s = lay.clone_state(st)
            logits, sem_f = lay.route(x, st_f, self.kmax, trace=True)
            m_short = self.rng.randint(1, self.kmax - 1)
            logits_s, sem_s = lay.route(x, st_s, m_short, trace=True)
            e_f = torch.softmax(logits, dim=-1)
            e_f[ar, tgt_col] = e_f[ar, tgt_col] - 1.0
            e_s = torch.softmax(logits_s, dim=-1)
            e_s[ar, tgt_col] = e_s[ar, tgt_col] - 1.0
            if mask is None:
                e_f = e_f / B
                e_s = e_s / B
                sc = None
            else:
                sc = (mask.unsqueeze(1) / mask.sum().clamp(min=1.0))
                e_f = e_f * sc
                e_s = e_s * sc
            g_ce = (e_s @ lay.head) * (sem_s * (1.0 - sem_s))
            g_cons = self.lam * (sem_s - sem_f.detach()) * (sem_s * (1.0 - sem_s))
            if sc is not None:
                g_cons = g_cons * sc
            lay.learn(sem_s, st_s, e=e_s, g_sem=g_ce + g_cons)
            lay.learn(sem_f, st_f, e=e_f)
            lay.copy_state_(st, st_f)
        else:
            k = self.kmax if self.mode == 'fixed' else self.rng.randint(1, self.kmax)
            logits, sem = lay.route(x, st, k, trace=True)
            e = torch.softmax(logits, dim=-1)
            e[ar, tgt_col] = e[ar, tgt_col] - 1.0
            if mask is None:
                e = e / B
            else:
                e = e * mask.unsqueeze(1) / mask.sum().clamp(min=1.0)
            lay.learn(sem, st, e=e)
        nll = torch.logsumexp(logits, dim=-1) - logits.gather(1, tgt_col.view(-1, 1)).squeeze(1)
        if mask is None:
            nll_sum = nll.sum()
            cnt = torch.tensor(float(B), device=DEV)
        else:
            nll_sum = (nll * mask).sum()
            cnt = mask.sum()
        return logits, nll_sum, cnt


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'r1'
    va_tok, va_tgt, va_msk = prep(va_ids, va_mask, NV, TV)

    def run(label, steps=STEPS, ks=(1, 2, 3, 4, 5, 6), **kw):
        steps = int(steps)
        tr_tok, tr_tgt, tr_msk = prep(tr_ids, tr_mask, NS, steps)
        net = ElasticNet(**kw)
        st = net.init_state(NS)
        t0 = time.time()
        for s in range(steps):
            net.step(tr_tok[:, s], tr_tgt[:, s], st, train=True, mask=tr_msk[:, s])
        dt = time.time() - t0
        scores = []
        for k in ks:
            stv = net.init_state(NV)
            tot, cnt = torch.zeros(()), torch.zeros(())
            for t in range(TV):
                lg, _, _ = net.step(va_tok[:, t], va_tgt[:, t], stv, train=False, K=k)
                nll = (torch.logsumexp(lg, dim=-1)
                       - lg.gather(1, va_tgt[:, t].view(-1, 1)).squeeze(1))
                tot = tot + (nll * va_msk[:, t]).sum()
                cnt = cnt + va_msk[:, t].sum()
            scores.append(float(tot / cnt.clamp(min=1.0)))
        print('[%s] mode=%s kmax=%d lam=%.1f h=%d params=%d %.1fs | %s'
              % (label, kw.get('mode'), kw.get('kmax'), kw.get('lam', 0),
                 kw.get('h', 192), net.n_params(), dt,
                 ' '.join('K%d=%.4f' % (k, c) for k, c in zip(ks, scores))))
        return scores

    if which == 'r1':
        run('fixedK4', mode='fixed', kmax=4)
        run('randomK', mode='random', kmax=4)
        run('consist', mode='consist', kmax=4, lam=1.0)
    if which == 'r2':
        run('cons-0.3', mode='consist', kmax=4, lam=0.3)
        run('cons-3.0', mode='consist', kmax=4, lam=3.0)
        run('consist-6000', mode='consist', kmax=4, lam=1.0, steps=6000)


if __name__ == '__main__':
    main()
