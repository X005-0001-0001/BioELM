"""循环+堆叠混合对照（expr-hybrid 分支，负结果留档）。

每层内部积分 loopk 次后层堆叠（hybrid） vs 纯循环 vs 纯堆叠。
用法（仓库根目录）：
    python experiments/bench_hybrid.py
环境变量 BIO_DEEP_SUBSET 可指定子集条数（CPU 建议 500，默认已设）。
协议 = deep_bench（CPU 干净 500 条子集，固定 η=2e-3，h=192，3000 步）。
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
torch.set_num_threads(min(16, os.cpu_count() or 8))

spec = importlib.util.spec_from_file_location('deepbench', os.path.join(BASE, 'experiments', 'deep_bench.py'))
DB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DB)

NS, NV, TV, STEPS = DB.NS, DB.NV, DB.TV, DB.STEPS


class HybridNet(DB.Net):
    def __init__(self, L=2, loopk=2, **kw):
        super().__init__('stacked', L=L, **kw)
        self.loopk = loopk

    @torch.no_grad()
    def step(self, tok_col, tgt_col, st, train=True, mask=None):
        x = self.E[tok_col]
        logits_list = []
        sems = []
        for i, lay in enumerate(self.layers):
            logits = sem = None
            for _ in range(self.loopk):
                logits, sem = lay.cell(x, st[i], trace=train)
            logits_list.append(logits)
            sems.append(sem)
            x = sem
        if not train:
            return logits_list[-1], None, None
        B = tok_col.numel()
        ar = torch.arange(B, device=DB.DEV)
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
        lg = logits_list[-1]
        nll = torch.logsumexp(lg, dim=-1) - lg.gather(1, tgt_col.view(-1, 1)).squeeze(1)
        if mask is None:
            nll_sum = nll.sum()
            cnt = torch.tensor(float(B), device=DB.DEV)
        else:
            nll_sum = (nll * mask).sum()
            cnt = mask.sum()
        return lg, nll_sum, cnt


va_tok, va_tgt, va_msk = DB.prep(DB.va_ids, DB.va_mask, NV, TV)
tr_tok, tr_tgt, tr_msk = DB.prep(DB.tr_ids, DB.tr_mask, NS, STEPS)


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
    print('[%s] params=%d %.1fs valCE=%.4f' % (label, net.n_params(), dt, ce))


run('single-192', DB.Net(mode='single', h=192))
run('loop2-192', DB.Net(mode='loop', K=2, h=192))
run('loop4-192', DB.Net(mode='loop', K=4, h=192))
run('stacked2-192', DB.Net(mode='stacked', L=2, h=192))
run('hybrid-L2K2', HybridNet(L=2, loopk=2))
run('hybrid-L2K1', HybridNet(L=2, loopk=1))
run('hybrid-L3K2', HybridNet(L=3, loopk=2))
