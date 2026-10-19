import math
import os
import time
import torch

DEV = torch.device(os.environ.get("BIO_DEV", "cpu"))


def _foreach_ok():
    try:
        a = [torch.zeros(2), torch.ones(3)]
        b = [torch.ones(2), torch.ones(3)]
        torch._foreach_mul_(a, 0.9)
        torch._foreach_add_(a, b, alpha=0.1)
        torch._foreach_addcmul_(a, b, b, value=0.5)
        m = torch._foreach_mul(a, 2.0)
        v = torch._foreach_sqrt(torch._foreach_div(a, 2.0))
        torch._foreach_add_(v, 1e-6)
        u = torch._foreach_div(m, v)
        torch._foreach_add_(a, u, alpha=-0.1)
        return True
    except Exception:
        return False


_FOREACH_OK = _foreach_ok()


class BioLSSM:
    def __init__(self, vocab, d=64, h=256, m=64, seed=43,
                 a_min=0.5, a_max=0.9999, learn_tau=True):
        g = torch.Generator().manual_seed(seed)
        self.vocab = vocab
        self.d = d
        self.h = h
        self.m = m
        self.learn_tau = learn_tau
        self.E = (torch.randn(vocab, d, generator=g) * 0.6).to(DEV)
        self.a_logit = torch.linspace(math.log(a_min / (1 - a_min + 1e-12)),
                                      math.log(a_max / (1 - a_max + 1e-12)),
                                      h).to(DEV)
        self.B = (torch.randn(h, d, generator=g) * (1.0 / math.sqrt(d))).to(DEV)
        self.Wp = ((torch.rand(m, h, generator=g) * 2 - 1) / math.sqrt(h)).to(DEV)
        self.bp = torch.zeros(m).to(DEV)
        self.head = ((torch.rand(vocab, m, generator=g) * 2 - 1) / math.sqrt(m)).to(DEV)
        self.bout = torch.zeros(vocab).to(DEV)
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.eta = 1e-3
        self.eps = 1e-6
        self.wd = 1e-4
        self.state = {k: (torch.zeros_like(p), torch.zeros_like(p), torch.zeros_like(p))
                      for k, p in BioLSSM.params(self).items()}
        self.steps = 0
        self._atp = torch.zeros((), device=DEV)
        self._atp_aux = 0.0
        self.train_params = None
        self._ar = {}
        # [O9] 偏差校正常量张量化（CUDA Graph 捕捉安全：值由 _set_bias_factors 在回放前写入）
        self._b1c = torch.ones((), device=DEV)
        self._b2c = torch.ones((), device=DEV)
        self._b1c_inv = torch.ones((), device=DEV)
        self._b2c_inv = torch.ones((), device=DEV)

    def _set_bias_factors(self):
        t = self.steps + 1
        b1c = 1.0 - self.beta1 ** t
        b2c = 1.0 - self.beta2 ** t
        self._b1c.fill_(b1c)
        self._b2c.fill_(b2c)
        self._b1c_inv.fill_(1.0 / b1c)
        self._b2c_inv.fill_(1.0 / b2c)

    def _snapshot_train_state(self):
        snap = {'steps': self.steps, 'atp': self._atp.clone(), 'atp_aux': self._atp_aux,
                'params': {k: v.detach().clone() for k, v in self.params().items()},
                'E': self.E.detach().clone(),
                'meta': {k: (m[0].clone(), m[1].clone()) for k, m in self.state.items()}}
        if getattr(self, 'learn_e', False):
            snap['E_state'] = (self.E_state[0].clone(), self.E_state[1].clone())
        return snap

    def _restore_train_state(self, snap):
        self.steps = snap['steps']
        self._atp.copy_(snap['atp'])
        self._atp_aux = snap['atp_aux']
        for k, v in self.params().items():
            v.copy_(snap['params'][k])
        self.E.copy_(snap['E'])
        for k, m in self.state.items():
            m[0].copy_(snap['meta'][k][0])
            m[1].copy_(snap['meta'][k][1])
        if 'E_state' in snap and getattr(self, 'learn_e', False):
            self.E_state[0].copy_(snap['E_state'][0])
            self.E_state[1].copy_(snap['E_state'][1])

    @property
    def atp(self):
        return float(self._atp) + self._atp_aux

    def _arange(self, B):
        t = self._ar.get(B)
        if t is None:
            t = torch.arange(B, device=DEV)
            self._ar[B] = t
        return t

    def _apply_grads(self, grads):
        items = []
        for name, grad in grads.items():
            if self.train_params is not None and name not in self.train_params:
                continue
            items.append((name, self.params()[name], grad))
        if not items:
            return
        b1c = self._b1c_inv   # [O9] 张量化偏差常量（图捕捉安全）
        b2c = self._b2c_inv
        names = [n for n, _, _ in items]
        params = [p for _, p, _ in items]
        gs = [g for _, _, g in items]
        mm = [self.state[n][0] for n in names]
        vv = [self.state[n][1] for n in names]
        if _FOREACH_OK:
            torch._foreach_mul_(mm, self.beta1)
            torch._foreach_add_(mm, gs, alpha=1.0 - self.beta1)
            torch._foreach_mul_(vv, self.beta2)
            torch._foreach_addcmul_(vv, gs, gs, value=1.0 - self.beta2)
            mhat = torch._foreach_mul(mm, b1c)
            vhat = torch._foreach_sqrt(torch._foreach_div(vv, self._b2c))
            torch._foreach_add_(vhat, self.eps)
            upd = torch._foreach_div(mhat, vhat)
            if self.wd > 0.0:
                p2 = [p for p in params if p.dim() == 2]
                if p2:
                    torch._foreach_mul_(p2, 1.0 - self.wd)
            torch._foreach_add_(params, upd, alpha=-self.eta)
            # [O8b] 批量 L1 范数（一次调用替代逐个 abs().sum()）
            try:
                s = sum(torch._foreach_norm(upd, ord=1))
            except Exception:
                s = upd[0].abs().sum()
                for u in upd[1:]:
                    s = s + u.abs().sum()
            self._atp.add_(s * 500.0)
        else:
            for name, param, grad in items:
                m, v, _ = self.state[name]
                m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
                mhat = m * b1c
                vhat = v / self._b2c
                upd = mhat / (vhat.sqrt() + self.eps)
                if self.wd > 0.0 and param.dim() == 2:
                    param.mul_(1.0 - self.wd)
                param.add_(upd, alpha=-self.eta)
                self._atp.add_(upd.abs().sum() * 500.0)
        self._atp_aux += sum(p.numel() for p in params) * 0.5

    def leak(self):
        return torch.sigmoid(self.a_logit)

    def params(self):
        p = {"a_logit": self.a_logit, "B": self.B, "Wp": self.Wp, "bp": self.bp,
             "head": self.head, "bout": self.bout}
        if not self.learn_tau:
            p.pop("a_logit")
        return p

    def n_params(self):
        return sum(t.numel() for t in self.params().values())

    def init_stream_state(self, batch):
        return {
            "s": torch.zeros(batch, self.h).to(DEV),
            "e_B": torch.zeros(batch, self.h, self.d).to(DEV),
            "e_a": torch.zeros(batch, self.h).to(DEV),
        }

    @torch.no_grad()
    def step_batch(self, tok, tgt, st, train=True, mask=None):
        B = tok.numel()
        x = self.E[tok]
        a = self.leak()
        s = a * st["s"] + x @ self.B.t()
        sem = torch.sigmoid(s @ self.Wp.t() + self.bp)
        logits = sem @ self.head.t() + self.bout
        st["e_B"] = a.view(1, -1, 1) * st["e_B"] + x.unsqueeze(1)
        st["e_a"] = a * st["e_a"] + st["s"]
        p = torch.softmax(logits, dim=-1)
        e = p
        e[self._arange(B), tgt] -= 1.0
        if mask is None:
            e = e / B
        else:
            m = mask.to(e.dtype).unsqueeze(1)
            e = e * m / m.sum().clamp(min=1.0)
        g_sem = (e @ self.head) * (sem * (1.0 - sem))
        Ls = g_sem @ self.Wp
        if train:
            grads = {}
            grads["a_logit"] = torch.einsum("bh,bh->h", Ls, st["e_a"]) * (a * (1.0 - a))
            grads["B"] = torch.einsum("bh,bhd->hd", Ls, st["e_B"])
            grads["Wp"] = g_sem.t() @ s
            grads["bp"] = g_sem.sum(0)
            grads["head"] = e.t() @ sem
            grads["bout"] = e.sum(0)
            if not torch.cuda.is_current_stream_capturing():
                self._set_bias_factors()   # 图捕捉时跳过（由 GraphStepRunner.run 在图外设置）
            self._apply_grads(grads)
            self.steps += 1
        st["s"] = s
        return logits

    @torch.no_grad()
    def eval_streams(self, streams, targets, crit_mask=None, mask=None):
        streams = streams.to(DEV)
        targets = targets.to(DEV)
        mask = mask.to(DEV) if mask is not None else None
        crit_mask = crit_mask.to(DEV) if crit_mask is not None else None
        n_streams, T = streams.shape
        st = self.init_stream_state(n_streams)
        tot = 0.0
        cnt = 0
        crit_tot = 0.0
        crit_cnt = 0
        for t in range(T):
            logits = self.step_batch(streams[:, t], targets[:, t], st, train=False)
            logp = torch.log_softmax(logits, dim=-1)
            nll = -logp[torch.arange(n_streams, device=logits.device), targets[:, t]]
            if mask is not None:
                mm = mask[:, t]
                tot += (nll * mm).sum().item()
                cnt += int(mm.sum().item())
            else:
                tot += nll.sum().item()
                cnt += n_streams
            if crit_mask is not None:
                m = crit_mask[:, t]
                if m.any():
                    crit_tot += nll[m].sum().item()
                    crit_cnt += int(m.sum().item())
        ce = tot / max(cnt, 1)
        cce = crit_tot / crit_cnt if crit_cnt else float("nan")
        return ce, cce

    @torch.no_grad()
    def train_streams(self, streams, targets, wall_seconds, log_every=30.0, log=print,
                      mask=None, use_graph=False, graph_runner=None):
        streams = streams.to(DEV)
        targets = targets.to(DEV)
        mask = mask.to(DEV) if mask is not None else None
        n_streams, T = streams.shape
        runner = graph_runner
        if runner is None and use_graph:
            runner = GraphStepRunner(self, n_streams)
        if runner is not None and not runner.ok:
            log("[LSSM] CUDA Graph 不可用，回退 eager：%s" % getattr(runner, 'err', '?'))
            runner = None
        elif runner is not None:
            log("[LSSM] CUDA Graph 训练步已启用（回放，免 Python 派发）")
            runner.reset_state()   # 与 eager 一致：每次 train_streams 以空状态开始
        st = None if runner is not None else self.init_stream_state(n_streams)
        t0 = time.time()
        last = t0
        steps = 0
        while True:
            for t in range(T):
                if runner is not None:
                    runner.run(streams[:, t], targets[:, t],
                               (mask[:, t] if mask is not None else None))
                else:
                    self.step_batch(streams[:, t], targets[:, t], st, train=True,
                                    mask=(mask[:, t] if mask is not None else None))
                steps += 1
                now = time.time()
                if now - last >= log_every:
                    log("[LSSM] t=%.0fs steps=%d tok/s=%.0f atp=%.2e%s"
                        % (now - t0, steps, steps * n_streams / (now - t0), self.atp,
                           ' (graph)' if runner is not None else ''))
                    last = now
                if now - t0 >= wall_seconds:
                    return steps, time.time() - t0
            if time.time() - t0 >= wall_seconds:
                return steps, time.time() - t0

    @torch.no_grad()
    def generate(self, prompt_ids, n_new=80, temp=0.8, top_p=0.9, seed=0):
        g = torch.Generator().manual_seed(seed)
        st = self.init_stream_state(1)
        logits = None
        for t in prompt_ids:
            logits = self.step_batch(torch.tensor([int(t)], device=DEV),
                                     torch.tensor([0], device=DEV),
                                     st, train=False)
        if logits is None:
            logits = self.step_batch(torch.tensor([0], device=DEV),
                                     torch.tensor([0], device=DEV), st,
                                     train=False)
        out = []
        for _ in range(n_new):
            z = logits[0].clone()
            z[2] = -1e9
            if temp <= 0.01:
                nxt = int(z.argmax())
            else:
                p = torch.softmax(z / temp, dim=-1)
                sp, si = torch.sort(p, descending=True)
                cs = torch.cumsum(sp, dim=0)
                mask = cs > top_p
                mask[0] = False
                sp[mask] = 0.0
                ssum = sp.sum()
                sp = sp / (ssum if ssum > 0 else 1.0)
                nxt = int(si[torch.multinomial(sp.cpu(), 1, generator=g)])
            out.append(nxt)
            logits = self.step_batch(torch.tensor([nxt], device=DEV),
                                     torch.tensor([0], device=DEV), st,
                                     train=False)
        return out


class BioGSSM(BioLSSM):
    def __init__(self, vocab, d=64, h=256, m=64, seed=43, learn_e=False, e_eta=1e-3,
                 trace_fp16=False, grad_einsum16=True):
        super().__init__(vocab, d=d, h=h, m=m, seed=seed, a_min=0.5, a_max=0.9999,
                         learn_tau=True)
        g = torch.Generator().manual_seed(seed + 1)
        self.W_g = ((torch.rand(h, d, generator=g) * 2 - 1) / math.sqrt(d)).to(DEV)
        self.b_g = torch.zeros(h).to(DEV)
        self.W_d = ((torch.rand(h, d, generator=g) * 2 - 1) / math.sqrt(d)).to(DEV)
        self.b_d = torch.full((h,), 3.0).to(DEV)
        for name in ("W_g", "b_g", "W_d", "b_d"):
            p = self.params()[name]
            self.state[name] = (torch.zeros_like(p), torch.zeros_like(p),
                                torch.zeros_like(p))
        self.state.pop("a_logit", None)   # GSSM 无 a_logit（随基类创建但从未使用，删除死状态）
        self.learn_e = bool(learn_e)
        self.e_eta = e_eta
        self.trace_fp16 = bool(trace_fp16)   # [O7] 资格迹 fp16 存储（fp32 归约），降带宽
        self.grad_einsum16 = bool(grad_einsum16)   # [O8a] 迹梯度用 fp16 einsum；False=旧 mul-sum(fp32 归约)
        if self.learn_e:
            self.Gf = ((torch.rand(h, d, generator=g) * 2 - 1)
                       / math.sqrt(h)).to(DEV)
            self.E_state = (torch.zeros_like(self.E), torch.zeros_like(self.E))
            self._gradE = torch.zeros_like(self.E)   # [O6] 复用缓冲，免逐步分配

    def params(self):
        p = super().params()
        p.pop("a_logit", None)
        p["W_g"] = self.W_g
        p["b_g"] = self.b_g
        p["W_d"] = self.W_d
        p["b_d"] = self.b_d
        return p

    def init_stream_state(self, batch):
        st = super().init_stream_state(batch)
        st.pop("e_B", None)   # GSSM 用融合迹 eK（(3,B,H,d)，首维连续），不再单独需要 e_B/e_a
        st.pop("e_a", None)
        tdt = torch.float16 if self.trace_fp16 else torch.float32
        st["eK"] = torch.zeros(3, batch, self.h, self.d, dtype=tdt).to(DEV)
        st["e_bg"] = torch.zeros(batch, self.h).to(DEV)
        st["e_bd"] = torch.zeros(batch, self.h).to(DEV)
        return st

    @torch.no_grad()
    def step_batch(self, tok, tgt, st, train=True, mask=None):
        B = tok.numel()
        x = self.E[tok]
        a = self.leak()
        g = torch.sigmoid(x @ self.W_g.t() + self.b_g)
        d = torch.sigmoid(x @ self.W_d.t() + self.b_d)
        u_raw = x @ self.B.t()
        u = g * u_raw
        # [O9] 状态就地推进（不重绑定张量 -> CUDA Graph 回放时状态可持续前进）
        s_buf = st["s"]
        c_g = (g * (1.0 - g)) * u_raw
        c_d = (d * (1.0 - d)) * s_buf        # 必须先于 s 就地更新计算
        s_buf.mul_(d).add_(u)
        s = s_buf
        sem = torch.sigmoid(s @ self.Wp.t() + self.bp)
        logits = sem @ self.head.t() + self.bout
        # [O5] 三条 (B,H,d) 资格迹融合为单张量 eK(3,B,H,d)：一次乘 + 一次 addcmul（原为 6 次算子）
        eK = st["eK"]
        if self.trace_fp16:
            # [O7] fp16 存储降带宽；更新算子同数学（fp16 舍入误差由 A/B 对照标定）
            eK.mul_(d.half().unsqueeze(-1).unsqueeze(0))
            eK.addcmul_(torch.stack((c_g, c_d, g), dim=0).half().unsqueeze(-1),
                        x.half().unsqueeze(0).unsqueeze(2))
        else:
            eK.mul_(d.unsqueeze(-1).unsqueeze(0))
            eK.addcmul_(torch.stack((c_g, c_d, g), dim=0).unsqueeze(-1),
                        x.unsqueeze(0).unsqueeze(2))
        st["e_bg"].mul_(d).add_(c_g)
        st["e_bd"].mul_(d).add_(c_d)
        p = torch.softmax(logits, dim=-1)
        e = p
        e[self._arange(B), tgt] -= 1.0
        if mask is None:
            e = e / B
        else:
            m = mask.to(e.dtype).unsqueeze(1)
            e = e * m / m.sum().clamp(min=1.0)
        g_sem = (e @ self.head) * (sem * (1.0 - sem))
        Ls = g_sem @ self.Wp
        if train:
            grads = {}
            if self.trace_fp16 and self.grad_einsum16:
                # [O8a] fp16 einsum 归约（内部 fp16 累加）：实测 79µs vs mul-sum 333µs（4.2×）
                Ls16 = Ls.half()
                grads["W_g"] = torch.einsum("bh,bhd->hd", Ls16, eK[0]).float()
                grads["W_d"] = torch.einsum("bh,bhd->hd", Ls16, eK[1]).float()
                grads["B"] = torch.einsum("bh,bhd->hd", Ls16, eK[2]).float()
            elif self.trace_fp16:
                # 旧路径：fp16 迹 -> 逐元素乘 + fp32 累加归约（更稳但更慢）
                Ls16 = Ls.half()
                grads["W_g"] = (Ls16.unsqueeze(-1) * eK[0]).sum(0, dtype=torch.float32)
                grads["W_d"] = (Ls16.unsqueeze(-1) * eK[1]).sum(0, dtype=torch.float32)
                grads["B"] = (Ls16.unsqueeze(-1) * eK[2]).sum(0, dtype=torch.float32)
            else:
                grads["W_g"] = torch.einsum("bh,bhd->hd", Ls, eK[0])
                grads["W_d"] = torch.einsum("bh,bhd->hd", Ls, eK[1])
                grads["B"] = torch.einsum("bh,bhd->hd", Ls, eK[2])
            grads["b_g"] = torch.einsum("bh,bh->h", Ls, st["e_bg"])
            grads["b_d"] = torch.einsum("bh,bh->h", Ls, st["e_bd"])
            grads["Wp"] = g_sem.t() @ s
            grads["bp"] = g_sem.sum(0)
            grads["head"] = e.t() @ sem
            grads["bout"] = e.sum(0)
            if not torch.cuda.is_current_stream_capturing():
                self._set_bias_factors()   # 图捕捉时跳过（由 GraphStepRunner.run 在图外设置）
            self._apply_grads(grads)
            if self.learn_e and self.train_params is None:
                delta = Ls @ self.Gf
                grad_e = self._gradE
                grad_e.zero_()
                grad_e.index_add_(0, tok, delta)
                mE, vE = self.E_state
                mE.mul_(self.beta1).add_(grad_e, alpha=1.0 - self.beta1)
                vE.mul_(self.beta2).addcmul_(grad_e, grad_e, value=1.0 - self.beta2)
                mhat = mE * self._b1c_inv
                vhat = vE / self._b2c
                Eupd = mhat / (vhat.sqrt() + self.eps)
                self.E.add_(Eupd, alpha=-self.e_eta)
                self._atp.add_(Eupd.abs().sum() * 200.0)   # [O9] 无 .item()，图捕捉安全
            self.steps += 1
        return logits


class GraphStepRunner:
    """[O9] 把一次训练步（含局部更新）抓成 CUDA Graph；回放免 Python 派发与启动开销。

    用法：
        runner = GraphStepRunner(model, n_streams)   # 失败时 runner.ok=False，调用方回退 eager
        runner.run(tok_col, tgt_col, mask_col)       # 每步一次；输入经 copy_ 进静态缓冲
    说明：
      - 捕捉前自动快照并还原训练状态（预热/抓取阶段的参数前进不回写模型）；流状态由 runner
        持有并随回放推进（reset_state() 可清零重开）。
      - run() 内部推进 model.steps 与偏差校正常量（张量化，图捕捉安全）。
      - 限制：step 内不得有 CPU 同步（本内核已无 .item()）；learn_e 关闭时最稳。
    """

    def __init__(self, model, ns, warmup=3):
        self.model = model
        self.ns = ns
        self.ok = False
        self.err = ''
        self.st = None
        self.out = None
        self.g = None
        try:
            if DEV.type != 'cuda':
                self.err = '非 CUDA 设备'
                return
            self.tok = torch.zeros(ns, dtype=torch.long, device=DEV)
            self.tgt = torch.zeros(ns, dtype=torch.long, device=DEV)
            self.mask = torch.ones(ns, device=DEV)
            snap = model._snapshot_train_state()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                st0 = model.init_stream_state(ns)
                for _ in range(warmup):
                    model._set_bias_factors()
                    model.step_batch(self.tok, self.tgt, st0, train=True, mask=self.mask)
                torch.cuda.synchronize()
            model._restore_train_state(snap)
            self.st = model.init_stream_state(ns)
            self.g = torch.cuda.CUDAGraph()
            # 注意：偏差常量必须在图外设置（放入图内会被回放覆盖为捕捉时的值）
            model._set_bias_factors()
            with torch.cuda.graph(self.g):
                self.out = model.step_batch(self.tok, self.tgt, self.st, train=True,
                                            mask=self.mask)
            model._restore_train_state(snap)
            for k in self.st:
                self.st[k].zero_()
            model._set_bias_factors()
            self.ok = True
        except Exception as e:
            self.ok = False
            self.err = str(e)[:160]

    def reset_state(self):
        if self.st is not None:
            for k in self.st:
                self.st[k].zero_()

    @torch.no_grad()
    def run(self, tok_col, tgt_col, mask_col):
        self.tok.copy_(tok_col)
        self.tgt.copy_(tgt_col)
        if mask_col is None:
            self.mask.fill_(1.0)
        else:
            if mask_col.dim() == 2:
                mask_col = mask_col[:, 0]
            self.mask.copy_(mask_col)
        self.model._set_bias_factors()
        self.g.replay()
        self.model.steps += 1
        return self.out
