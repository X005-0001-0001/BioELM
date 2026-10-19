"""学习型停机（learned halting）探索：
小读出头 p_halt = σ(sem_k·w + b)，用三因子/逻辑回归本地规则在线训练，
目标 = "下一步边际 CE 不再改善"（自监督，零人工阈值）：y_k = 1[CE_k ≤ CE_{k+1}]，y_last = 1。
推理：逐子步运行，p_halt > 0.5 即停（≤KMAX）。
协议：仓库 bench 同口径（干净 500 条子集，η=2e-3，h=192，consist KMAX=4，3000 步）。"""
import os
import sys
import math
import time
import importlib.util

os.environ['BIO_DEV'] = 'cpu'
os.environ.setdefault('BIO_ELASTIC_SUBSET', '500')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import torch
torch.set_num_threads(min(16, os.cpu_count() or 8))

DEV = torch.device('cpu')
VOCAB = 2754
ETA = 2.0e-3
WD = 1.0e-4
BETA1, BETA2, EPS = 0.9, 0.99, 1e-6
KMAX = 4
ETA_H = 0.1


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = load(os.path.join(BASE, 'bioelm_elite_1.1.5.py'), 'b115')
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('bench_el', os.path.join(BASE, 'experiments', 'bench_elastic.py'))
_B = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_B)
tr_ids, tr_mask, va_ids, va_mask = _B.tr_ids, _B.tr_mask, _B.va_ids, _B.va_mask

NS, NV, TV, STEPS = 32, 8, 200, 3000


def prep(ids, mask, ns, T):
    need = ns * T + 1
    return (torch.tensor(ids[:need - 1], dtype=torch.long).view(ns, T),
            torch.tensor(ids[1:need], dtype=torch.long).view(ns, T),
            torch.tensor(mask[1:need], dtype=torch.float32).view(ns, T))


tr_tok, tr_tgt, tr_msk = prep(tr_ids, tr_mask, NS, STEPS)
va_tok, va_tgt, va_msk = prep(va_ids, va_mask, NV, TV)


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

    @torch.no_grad()
    def route(self, x, st, K, trace=True, collect=False):
        logits = sem = None
        lg_list, sem_list = [], []
        for _ in range(K):
            logits, sem = self.cell(x, st, trace=trace)
            if collect:
                lg_list.append(logits)
                sem_list.append(sem.detach())
        return logits, sem, lg_list, sem_list


class Net:
    def __init__(self, lam=1.0, h=192, m=48, d=48, seed=43, rng_seed=1234):
        import random
        self.lam = lam
        self.rng = random.Random(rng_seed)
        gen = torch.Generator().manual_seed(seed)
        self.E = (torch.randn(VOCAB, d, generator=gen) * 0.6).to(DEV)
        self.layer = Layer(d, h, m, gen)
        self.hw = torch.zeros(m).to(DEV)
        self.hb = torch.zeros(()).to(DEV)

    def init_state(self, batch):
        return self.layer.init_state(batch)

    @torch.no_grad()
    def halt_prob(self, sem):
        return torch.sigmoid(sem @ self.hw + self.hb)

    @torch.no_grad()
    def train_halter(self, lg_list, sem_list, tgt, mask):
        ce = []
        for lg in lg_list:
            nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, tgt.view(-1, 1)).squeeze(1)
            ce.append(nll.detach())
        K = len(ce)
        pairs_sem, pairs_y = [], []
        for k in range(K - 1):
            y = (ce[k] <= ce[k + 1]).float()
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
        self.hw -= ETA_H * (err @ sem_all)
        self.hb -= ETA_H * err.sum()

    @torch.no_grad()
    def step(self, tok_col, tgt_col, st, train=True, mask=None, K_eval=None):
        lay = self.layer
        x = self.E[tok_col]
        if not train:
            k = KMAX if K_eval is None else K_eval
            logits, sem, _, _ = lay.route(x, st, k, trace=False)
            return logits, None, None
        Bn = tok_col.numel()
        ar = torch.arange(Bn, device=DEV)
        st_f = lay.clone_state(st)
        st_s = lay.clone_state(st)
        logits, sem_f, lg_list, sem_list = lay.route(x, st_f, KMAX, trace=True, collect=True)
        m_short = self.rng.randint(1, KMAX - 1)
        logits_s, sem_s, _, _ = lay.route(x, st_s, m_short, trace=True)
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
        lay.learn(sem_s, st_s, e=e_s, g_sem=g_ce + g_cons)
        lay.learn(sem_f, st_f, e=e_f)
        lay.copy_state_(st, st_f)
        self.train_halter(lg_list, sem_list, tgt_col, mask)
        return logits, None, None


# 复用 repo bench 的 learn/apply（此处 Layer 缺 learn/apply，直接补上，从 bench_elastic 复制）
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('bench_el', os.path.join(BASE, 'experiments', 'bench_elastic.py'))
_B = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_B)
Layer.learn = _B.Layer.learn
Layer.apply = _B.Layer.apply


def evaluate_fixed(net, K):
    stv = net.init_state(NV)
    tot, cnt = torch.zeros(()), torch.zeros(())
    for t in range(TV):
        lg, _, _ = net.step(va_tok[:, t], va_tgt[:, t], stv, train=False, K_eval=K)
        nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, va_tgt[:, t].view(-1, 1)).squeeze(1)
        tot = tot + (nll * va_msk[:, t]).sum()
        cnt = cnt + va_msk[:, t].sum()
    return float(tot / cnt.clamp(min=1.0))


def evaluate_halt(net):
    stv = net.init_state(NV)
    tot, cnt = torch.zeros(()), torch.zeros(())
    depth_sum, depth_n = 0.0, 0
    for t in range(TV):
        x = net.E[va_tok[:, t]]
        lay = net.layer
        lg = sem = None
        used = KMAX
        for k in range(KMAX):
            lg, sem = lay.cell(x, stv, trace=False)
            if k >= 1 and float(net.halt_prob(sem).mean()) > 0.5:
                used = k + 1
                break
        depth_sum += used * x.shape[0]
        depth_n += x.shape[0]
        nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, va_tgt[:, t].view(-1, 1)).squeeze(1)
        tot = tot + (nll * va_msk[:, t]).sum()
        cnt = cnt + va_msk[:, t].sum()
    return float(tot / cnt.clamp(min=1.0)), depth_sum / depth_n


net = Net()
st = net.init_state(NS)
t0 = time.time()
for s in range(STEPS):
    net.step(tr_tok[:, s], tr_tgt[:, s], st, train=True, mask=tr_msk[:, s])
dt = time.time() - t0
print('trained %.1fs' % dt)
print('fixed: ' + ' '.join('K%d=%.4f' % (k, evaluate_fixed(net, k)) for k in (1, 2, 3, 4)))
ce, d = evaluate_halt(net)
print('auto-halt valCE=%.4f avgDepth=%.3f' % (ce, d))
