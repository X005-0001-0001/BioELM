"""struct_lab_bd —— 分块对角学习式通道混合（tour-blockdiag 分支）。

状态按 G 组、每组 s 维；组内用可学矩阵 A_g 混合状态：
    S' = d ⊙ (S A) + g ⊙ u
A_g 初始化为 I → 起点与基线（对角）逐位一致；随后学习局部混合。
账本：eK/偏置迹改为先乘 A 再衰减；新增 eA（参数 A 的精确迹，≈ s²/(3d) 的 eK 规模，+11%）。
"""
import os
import sys
import math
import time
import importlib.util

os.environ.setdefault('BIO_DEV', 'cuda')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

import torch

DEV = torch.device(os.environ.get('BIO_DEV', 'cuda'))
torch.set_num_threads(min(8, os.cpu_count() or 8))

CFG = {
    'K': 2,
    'BD': 4,
    'BD_FREEZE': 0,
    'h': 192,
    'd': 48,
    'm': 48,
    'streams': 1024,
    'steps': 40000,
    'checkpoints': (10000, 40000),
    'eta': 2.0e-3,
    'wd': 1.0e-4,
    'b1': 0.9,
    'b2': 0.99,
    'eps': 1e-6,
    'seed': 43,
    'TAU': 1.0,
}

NV, TV = 128, 200
CHUNK = 400

K = CFG['K']
BD = CFG['BD']
H, D, M = CFG['h'], CFG['d'], CFG['m']
SEED = CFG['seed']
G = H // BD


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = load(os.path.join(BASE, 'bioelm_elastic_1.1.1.py'), 'bd_loader')
P.LOG_FILE = os.path.join(os.environ.get('TEMP', '.'), 'struct_lab_bd.log')
P.init_log_file()
tok, _ = P._load_any_tokenizer()
VOCAB = tok.vocab_size
DATA = os.environ.get('BIO_DEEP_DATA', os.path.join(BASE, 'dsr1_v5_0001.jsonl'))
tr_ids, tr_mask, va_ids, va_mask = P.build_corpus(tok, DATA)


class Net:
    def __init__(self):
        gen = torch.Generator().manual_seed(SEED)
        self.E = (torch.randn(VOCAB, D, generator=gen) * 0.6).to(DEV)
        self.W_g = ((torch.rand(H, D, generator=gen) * 2 - 1) / math.sqrt(D)).to(DEV)
        self.b_g = torch.zeros(H, device=DEV)
        self.W_d = ((torch.rand(H, D, generator=gen) * 2 - 1) / math.sqrt(D)).to(DEV)
        d0 = 1.0 - math.log(2.0) / (CFG['TAU'] * math.sqrt(H))
        self.b_d = torch.full((H,), math.log(d0 / (1.0 - d0)), device=DEV)
        self.B = (torch.randn(H, D, generator=gen) / math.sqrt(D)).to(DEV)
        self.Wp = ((torch.rand(M, H, generator=gen) * 2 - 1) / math.sqrt(H)).to(DEV)
        self.bp = torch.zeros(M, device=DEV)
        self.head = ((torch.rand(VOCAB, M, generator=gen) * 2 - 1) / math.sqrt(M)).to(DEV)
        self.bout = torch.zeros(VOCAB, device=DEV)
        self.A = torch.eye(BD, device=DEV).repeat(G, 1, 1)
        self.eye = torch.eye(BD, device=DEV)
        self.t = 0
        self._meta = {k: (torch.zeros_like(v), torch.zeros_like(v))
                      for k, v in self.params().items()}

    def params(self):
        return {'W_g': self.W_g, 'b_g': self.b_g, 'W_d': self.W_d, 'b_d': self.b_d,
                'B': self.B, 'Wp': self.Wp, 'bp': self.bp, 'head': self.head,
                'bout': self.bout, 'A': self.A}

    def n_params(self):
        return sum(v.numel() for v in self.params().values()) + self.E.numel()

    def init_state(self, batch):
        return {
            's': torch.zeros(batch, H, device=DEV),
            'eK': torch.zeros(3, batch, H, D, device=DEV),
            'e_bg': torch.zeros(batch, H, device=DEV),
            'e_bd': torch.zeros(batch, H, device=DEV),
            'eA': torch.zeros(batch, G, BD, BD, BD, device=DEV),
        }

    @torch.no_grad()
    def _cell(self, x, st, trace):
        g = torch.sigmoid(x @ self.W_g.t() + self.b_g)
        d = torch.sigmoid(x @ self.W_d.t() + self.b_d)
        u_raw = x @ self.B.t()
        u = g * u_raw
        s = st['s']
        Sg = s.view(-1, G, BD)
        mixed = torch.einsum('bgs,gsk->bgk', Sg, self.A).reshape(-1, H)
        if trace:
            c_g = (g * (1.0 - g)) * u_raw
            c_d = (d * (1.0 - d)) * mixed
        s.copy_(mixed.mul(d).add(u))
        sem = torch.sigmoid(s @ self.Wp.t() + self.bp)
        logits = sem @ self.head.t() + self.bout
        if trace:
            eK = st['eK'].view(3, -1, G, BD, D)
            eK.copy_(torch.einsum('cbgsd,gsk->cbgkd', eK, self.A).mul(d.view(-1, G, BD).unsqueeze(0).unsqueeze(-1)))
            st['eK'].addcmul_(torch.stack((c_g, c_d, g), 0).unsqueeze(-1), x.unsqueeze(0).unsqueeze(2))
            st['e_bg'].copy_(torch.einsum('bgs,gsk->bgk', st['e_bg'].view(-1, G, BD), self.A).reshape(-1, H).mul(d).add(c_g))
            st['e_bd'].copy_(torch.einsum('bgs,gsk->bgk', st['e_bd'].view(-1, G, BD), self.A).reshape(-1, H).mul(d).add(c_d))
            eA = st['eA']
            eA.copy_(torch.einsum('bgijk,gkc->bgijc', eA, self.A)
                     .mul(d.view(-1, G, BD).unsqueeze(2).unsqueeze(2))
                     + Sg.unsqueeze(-1).unsqueeze(-1) * d.view(-1, G, 1, 1, BD)
                     * self.eye[None, None, :, :, None])
        return logits, sem

    @torch.no_grad()
    def step(self, tok_col, tgt_col, st, train=True, mask=None):
        x = self.E[tok_col]
        for _ in range(K):
            logits, sem = self._cell(x, st, trace=train)
        if not train:
            return logits
        Bn = tok_col.numel()
        ar = torch.arange(Bn, device=DEV)
        p = torch.softmax(logits, dim=-1)
        e = p
        e[ar, tgt_col] = e[ar, tgt_col] - 1.0
        if mask is None:
            e = e / Bn
        else:
            mm = mask.to(e.dtype).unsqueeze(1)
            e = e * mm / mm.sum().clamp(min=1.0)
        self._learn(sem, st, e)
        return logits

    @torch.no_grad()
    def _learn(self, sem, st, e):
        g_sem = (e @ self.head) * (sem * (1.0 - sem))
        Ls = g_sem @ self.Wp
        eK = st['eK']
        grads = [
            ('W_g', torch.einsum('bh,bhd->hd', Ls, eK[0])),
            ('b_g', torch.einsum('bh,bh->h', Ls, st['e_bg'])),
            ('W_d', torch.einsum('bh,bhd->hd', Ls, eK[1])),
            ('b_d', torch.einsum('bh,bh->h', Ls, st['e_bd'])),
            ('B', torch.einsum('bh,bhd->hd', Ls, eK[2])),
            ('Wp', g_sem.t() @ st['s']),
            ('bp', g_sem.sum(0)),
            ('head', e.t() @ sem),
            ('bout', e.sum(0)),
        ]
        if not CFG['BD_FREEZE']:
            grads.append(('A', torch.einsum('bgc,bgijc->gij', Ls.view(-1, G, BD), st['eA'])))
        self._apply(grads)

    @torch.no_grad()
    def _apply(self, grads):
        self.t += 1
        b1c = 1.0 - CFG['b1'] ** self.t
        b2c = 1.0 - CFG['b2'] ** self.t
        names = [n for n, _ in grads]
        gs = [g for _, g in grads]
        ps = [self.params()[n] for n in names]
        mm = [self._meta[n][0] for n in names]
        vv = [self._meta[n][1] for n in names]
        torch._foreach_mul_(mm, CFG['b1'])
        torch._foreach_add_(mm, gs, alpha=1.0 - CFG['b1'])
        torch._foreach_mul_(vv, CFG['b2'])
        torch._foreach_addcmul_(vv, gs, gs, value=1.0 - CFG['b2'])
        mhat = torch._foreach_mul(mm, 1.0 / b1c)
        vhat = torch._foreach_sqrt(torch._foreach_div(vv, b2c))
        torch._foreach_add_(vhat, CFG['eps'])
        upd = torch._foreach_div(mhat, vhat)
        p2 = [p for p in ps if p.dim() == 2]
        if p2:
            torch._foreach_mul_(p2, 1.0 - CFG['wd'])
        torch._foreach_add_(ps, upd, alpha=-CFG['eta'])


def to_batches(ids, mask, ns, T, rotate):
    s = torch.tensor(ids[rotate:rotate + ns * T + 1])
    st = s[:-1].view(ns, T).t().contiguous().to(DEV)
    tg = s[1:].view(ns, T).t().contiguous().to(DEV)
    mk = torch.tensor(mask[rotate + 1:rotate + ns * T + 1], dtype=torch.float32).view(ns, T).t().contiguous().to(DEV)
    return st, tg, mk


@torch.no_grad()
def evaluate(net, ids, mask, ns, T):
    s = torch.tensor(ids[:ns * T + 1], dtype=torch.long)
    tt = s[:-1].view(ns, T)
    tg = s[1:].view(ns, T)
    mm = torch.tensor(mask[1:ns * T + 1], dtype=torch.float32).view(ns, T)
    st = net.init_state(ns)
    tot = torch.zeros((), device=DEV)
    cnt = torch.zeros((), device=DEV)
    for t in range(T):
        tok = tt[:, t].to(DEV)
        tgt = tg[:, t].to(DEV)
        mk = mm[:, t].to(DEV)
        lg = net.step(tok, tgt, st, train=False)
        nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, tgt.view(-1, 1)).squeeze(1)
        tot = tot + (nll * mk).sum()
        cnt = cnt + mk.sum()
    return float(tot / cnt.clamp(min=1.0))


def run(cfg):
    global K, SEED
    CFG.update(cfg)
    K, SEED = CFG['K'], CFG['seed']
    net = Net()
    st = net.init_state(CFG['streams'])
    n = CFG['streams']
    marks = sorted(set(CFG['checkpoints']))
    out = {}
    done = 0
    rot = 0
    t0 = time.time()
    while done < CFG['steps']:
        bt, bg, bm = to_batches(tr_ids, tr_mask, n, CHUNK, rot)
        for s in range(bt.shape[0]):
            net.step(bt[s], bg[s], st, train=True, mask=bm[s])
            done += 1
            if done in marks:
                out[done] = evaluate(net, va_ids, va_mask, NV, TV)
                print('  step %d: valCE=%.4f ppl=%.2f (%.0fs)'
                      % (done, out[done], math.exp(out[done]), time.time() - t0), flush=True)
            if done >= CFG['steps']:
                break
        rot = rot + n * CHUNK + 1
        if rot + n * CHUNK + 2 >= len(tr_ids):
            rot = 0
    dt = time.time() - t0
    print('[BD=%d freeze=%d K=%d TAU=%.1f seed=%d] %d steps %.1fs (%.1f st/s) params=%d Adev=%.4f'
          % (BD, CFG['BD_FREEZE'], K, CFG['TAU'], SEED, CFG['steps'], dt, CFG['steps'] / dt,
             net.n_params(), float((net.A - net.eye[None]).abs().max())), flush=True)
    return out


if __name__ == '__main__':
    run({})
