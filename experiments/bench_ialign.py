"""层间接口对齐（learned-alignment FA）探索：
两层堆叠，层间变换 x2 = sem1 @ W_l（可学，恒等初始化），
- W_l 的梯度：sem1ᵀ @ gx2（gx2 = 对 layer2 输入的局部梯度，解析计算，无回传）
- layer1 的学习信号 = (gx2 @ W_lᵀ) ⊙ sem1(1-sem1)——把随机回投矩阵 F 换成"可学对齐"的直接梯度
对照：纯堆叠 L2（各层自带头）、随机 FA L2。协议 = deep_bench（CPU 500 条，eta=2e-3，h=192，3000 步）。
"""
import os
import sys
import time
import importlib.util

os.environ.setdefault('BIO_DEV', 'cpu')
os.environ.setdefault('BIO_DEEP_SUBSET', '500')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

import torch
torch.set_num_threads(8)

spec = importlib.util.spec_from_file_location('deepbench', os.path.join(BASE, 'experiments', 'deep_bench.py'))
DB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DB)

NS, NV, TV, STEPS = DB.NS, DB.NV, DB.TV, DB.STEPS
va_tok, va_tgt, va_msk = DB.prep(DB.va_ids, DB.va_mask, NV, TV)
tr_tok, tr_tgt, tr_msk = DB.prep(DB.tr_ids, DB.tr_mask, NS, STEPS)


class IAlignNet:
    def __init__(self, h=192, m=48, d=48, seed=43):
        gen = torch.Generator().manual_seed(seed)
        self.E = (torch.randn(DB.VOCAB, d, generator=gen) * 0.6).to(DB.DEV)
        self.lay1 = DB.Layer(d, h, m, gen, with_head=False)
        self.lay2 = DB.Layer(m, h, m, gen, with_head=True)
        self.W_l = (torch.eye(m) + 0.01 * torch.randn(m, m, generator=gen)).to(DB.DEV)
        self.lay2.W_l = self.W_l
        _op = self.lay2.params

        def params_with_link():
            dd = _op()
            dd['W_l'] = self.W_l
            return dd

        self.lay2.params = params_with_link
        self.lay2._meta['W_l'] = (torch.zeros_like(self.W_l), torch.zeros_like(self.W_l))

    def n_params(self):
        return sum(p.numel() for p in self.lay1.params().values()) + \
               sum(p.numel() for p in self.lay2.params().values()) + self.E.numel()

    def init_state(self, batch):
        return [self.lay1.init_state(batch), self.lay2.init_state(batch)]

    @torch.no_grad()
    def step(self, tok_col, tgt_col, st, train=True, mask=None):
        x1 = self.E[tok_col]
        logits1, sem1 = self.lay1.cell(x1, st[0], trace=train)
        x2 = sem1 @ self.lay2.W_l
        s_prev = st[1]['s'].clone()
        logits2, sem2 = self.lay2.cell(x2, st[1], trace=train)
        if not train:
            return logits2, None, None
        B = tok_col.numel()
        ar = torch.arange(B, device=DB.DEV)
        e = torch.softmax(logits2, dim=-1)
        e[ar, tgt_col] = e[ar, tgt_col] - 1.0
        scale = (torch.ones(B, device=DB.DEV) / B) if mask is None else (mask / mask.sum().clamp(min=1.0))
        e_sc = e * scale.unsqueeze(1)
        g_sem2 = (e_sc @ self.lay2.head) * (sem2 * (1.0 - sem2))
        Ls2 = g_sem2 @ self.lay2.Wp
        # gx2：对 layer2 输入的局部解析梯度（三层贡献：d 门、g 门、原始输入）
        g2 = torch.sigmoid(x2 @ self.lay2.W_g.t() + self.lay2.b_g)
        d2 = torch.sigmoid(x2 @ self.lay2.W_d.t() + self.lay2.b_d)
        u_raw2 = x2 @ self.lay2.B.t()
        c_g2 = (g2 * (1.0 - g2)) * u_raw2
        c_d2 = (d2 * (1.0 - d2)) * s_prev
        gx2 = (torch.einsum('bh,hd->bd', Ls2 * c_d2, self.lay2.W_d)
               + torch.einsum('bh,hd->bd', Ls2 * g2, self.lay2.B)
               + torch.einsum('bh,hd->bd', Ls2 * c_g2, self.lay2.W_g))
        g_Wl = torch.einsum('bm,bd->md', sem1, gx2)
        g_sem1 = (gx2 @ self.lay2.W_l.t()) * (sem1 * (1.0 - sem1))
        # layer2 学习（CE 信号） + W_l
        grads2 = [g for g in self.lay2._grads_from(sem2, st[1], e_sc)]
        grads2 += [('W_l', g_Wl)]
        self.lay2.apply(grads2)
        # layer1 学习（对齐回投信号）
        self.lay1.learn(sem1, st[0], g_sem=g_sem1)
        nll = torch.logsumexp(logits2, dim=-1) - logits2.gather(1, tgt_col.view(-1, 1)).squeeze(1)
        if mask is None:
            nll_sum = nll.sum()
            cnt = torch.tensor(float(B), device=DB.DEV)
        else:
            nll_sum = (nll * mask).sum()
            cnt = mask.sum()
        return logits2, nll_sum, cnt


def _grads_from(self, sem, st, e=None, g_sem=None):
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
    if self.head is not None and e is not None:
        grads += [('head', e.t() @ sem), ('bout', e.sum(0))]
    return grads


DB.Layer._grads_from = _grads_from


def run(label, net):
    st = net.init_state(NS)
    t0 = time.time()
    for s in range(STEPS):
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
    print('[%s] params=%d %.1fs valCE=%.4f' % (label, net.n_params(), dt, ce), flush=True)


run('ialign-L2', IAlignNet())
run('fa-random-L2', DB.Net(mode='fa', L=2, h=192))
run('stacked-L2', DB.Net(mode='stacked', L=2, h=192))
print('DONE')
