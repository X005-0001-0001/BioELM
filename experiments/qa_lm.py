import argparse
import array
import json
import math
import os
import sys
import time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_num_threads(min(16, os.cpu_count() or 1))
DEV = torch.device(os.environ.get("BIO_DEV", "cpu"))

import data2 as D
from biolssm import BioGSSM
from gtf2 import TinyGPT2


def load_ids(path):
    a = array.array("i")
    with open(path, "rb") as f:
        a.fromfile(f, os.path.getsize(path) // 4)
    return list(a)


def load_mask(path):
    a = array.array("B")
    with open(path, "rb") as f:
        a.fromfile(f, os.path.getsize(path))
    return list(a)


@torch.no_grad()
def bio_eval_masked(rnn, ids, mask, n_streams=8):
    st, tg, _ = D.to_streams(ids, D.shift_next(ids), [0] * len(ids), n_streams)
    mk = torch.tensor(mask, dtype=torch.float)
    mk = mk[-st.numel():].reshape(n_streams, -1)
    ce_m, _ = rnn.eval_streams(st, tg, None, mask=mk)
    return ce_m


@torch.no_grad()
def tf_eval_masked(model, ids, mask, k, batch=256):
    n = len(ids) - 1 - k
    tot = 0.0
    cnt = 0
    model.eval()
    dev = next(model.parameters()).device
    for s in range(0, n, batch):
        e = min(n, s + batch)
        xs = torch.stack([torch.tensor(ids[i:i + k], dtype=torch.long)
                          for i in range(s, e)]).to(dev)
        ys = torch.tensor([ids[i + k] for i in range(s, e)]).to(dev)
        mk = torch.tensor([mask[i + k] for i in range(s, e)], dtype=torch.bool).to(dev)
        logits = model(xs)
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp[torch.arange(e - s, device=dev), ys]
        if mk.any():
            tot += nll[mk].sum().item()
            cnt += int(mk.sum().item())
    return tot / max(cnt, 1)


def tf_train_masked(model, flat_x, flat_mask, wall_seconds, lr=1e-3, batch=64,
                    log_every=30.0, log=print):
    k = model.k
    n = len(flat_x)
    dev = next(model.parameters()).device
    flat_t = torch.tensor(flat_x, dtype=torch.long)
    active = [i for i in range(n - k) if flat_mask[i + k]]
    active_t = torch.tensor(active, dtype=torch.long)
    g = torch.Generator().manual_seed(1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time()
    last = t0
    steps = 0
    while True:
        idx = active_t[torch.randint(0, active_t.numel(), (batch,), generator=g)]
        xs = torch.stack([flat_t[i:i + k] for i in idx.tolist()]).to(dev)
        ys = torch.tensor([flat_t[int(i) + k] for i in idx.tolist()]).to(dev)
        logits = model(xs)
        loss = torch.nn.functional.cross_entropy(logits, ys)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps += 1
        now = time.time()
        if now - last >= log_every:
            log("[QA-tf] t=%.0fs steps=%d tok/s=%.0f loss=%.4f"
                % (now - t0, steps, steps * batch / (now - t0), loss.item()))
            last = now
        if now - t0 >= wall_seconds:
            return steps, time.time() - t0


def unigram_ppl(tr_ids, tr_mask, va_ids, va_mask, vocab):
    cnt = torch.zeros(vocab)
    for t, m in zip(tr_ids, tr_mask):
        if m:
            cnt[t] += 1.0
    p = cnt / cnt.sum()
    tot = 0.0
    n = 0
    for t, m in zip(va_ids, va_mask):
        if m:
            tot += -math.log(max(p[t].item(), 1e-12))
            n += 1
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default=os.path.join(HERE, "datasets"))
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--h", type=int, default=384)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--eta", type=float, default=3e-3)
    ap.add_argument("--only", default="both")
    ap.add_argument("--tf-ctx", type=int, default=128)
    ap.add_argument("--tf-lr", type=float, default=1e-3)
    ap.add_argument("--ckpt", default="ckpt_gssm_qa.pt")
    ap.add_argument("--learn-e", type=int, default=0)
    ap.add_argument("--e-eta", type=float, default=1e-3)
    ap.add_argument("--trace-fp16", type=int, default=0)
    ap.add_argument("--graph", type=int, default=0)
    ap.add_argument("--grad-einsum16", type=int, default=1)
    ap.add_argument("--log", default=os.path.join(HERE, "logs", "qa_lm.log"))
    args = ap.parse_args()

    logf = open(args.log, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    meta = json.load(open(os.path.join(args.datadir, "qa_meta.json"),
                          encoding="utf-8"))
    vocab = meta["vocab"]
    tr_ids = load_ids(os.path.join(args.datadir, "qa_train.ids.bin"))
    tr_mask = load_mask(os.path.join(args.datadir, "qa_train.mask.bin"))
    va_ids = load_ids(os.path.join(args.datadir, "qa_val.ids.bin"))
    va_mask = load_mask(os.path.join(args.datadir, "qa_val.mask.bin"))
    log("=" * 78)
    log("[QA] vocab=%d train=%d(learn %d) val=%d(learn %d)"
        % (vocab, len(tr_ids), sum(tr_mask), len(va_ids), sum(va_mask)))

    ce_uni = unigram_ppl(tr_ids, tr_mask, va_ids, va_mask, vocab)
    log("[QA] unigram baseline: val masked ce=%.4f ppl=%.2f"
        % (ce_uni, math.exp(min(ce_uni, 20))))

    st, tg, _ = D.to_streams(tr_ids, D.shift_next(tr_ids), [0] * len(tr_ids),
                             args.streams)
    mk = torch.tensor(tr_mask, dtype=torch.float)
    mk = mk[-st.numel():].reshape(args.streams, -1)
    log("[QA] streams=%d T=%d" % (args.streams, st.shape[1]))

    sub = 40000
    if args.only in ("both", "bio"):
        torch.manual_seed(43)
        rnn = BioGSSM(vocab, d=args.d, h=args.h, m=args.m, seed=43,
                      learn_e=bool(args.learn_e), e_eta=args.e_eta,
                      trace_fp16=bool(args.trace_fp16),
                      grad_einsum16=bool(args.grad_einsum16))
        rnn.eta = args.eta
        log("[QA-bio] params=%d h=%d eta=%.0e dev=%s learn_e=%d e_eta=%.0e"
            % (rnn.n_params(), args.h, rnn.eta, str(DEV), args.learn_e,
               args.e_eta))
        n_chunks = max(1, int(args.budget // 60))
        chunk = args.budget / n_chunks
        runner = None
        if args.graph:
            from biolssm import GraphStepRunner
            runner = GraphStepRunner(rnn, args.streams)
            log("[QA-bio] CUDA Graph runner ok=%s %s" % (runner.ok, runner.err))
            if not runner.ok:
                runner = None
        dt = 0.0
        steps = 0
        for _ in range(n_chunks):
            s2, d2 = rnn.train_streams(st, tg, chunk, log=log, mask=mk,
                                       graph_runner=runner)
            steps += s2
            dt += d2
            ce_sub = bio_eval_masked(rnn, va_ids[-sub:], va_mask[-sub:])
            log("[QA-bio] mid t=%.0fs steps=%d val_sub masked_ce=%.4f ppl=%.2f"
                % (dt, steps, ce_sub, math.exp(min(ce_sub, 20))))
        t0 = time.time()
        ce_v = bio_eval_masked(rnn, va_ids, va_mask)
        log("[QA-bio] FINAL val masked_ce=%.4f ppl=%.2f | steps=%d tps=%.0f "
            "atp=%.2e eval=%.0fs"
            % (ce_v, math.exp(min(ce_v, 20)), steps,
               steps * args.streams / dt, rnn.atp, time.time() - t0))
        ck = {k: v.detach().cpu().clone() for k, v in rnn.params().items()}
        ck["E"] = rnn.E.detach().cpu().clone()
        torch.save(ck, os.path.join(HERE, args.ckpt))

    if args.only in ("both", "tf"):
        torch.manual_seed(43)
        tfm = TinyGPT2(vocab, k=args.tf_ctx, d=64, layers=1, ff=128).to(DEV)
        log("[QA-tf] params=%d ctx=%d dev=%s"
            % (tfm.n_params(), args.tf_ctx, DEV.type))
        steps, dt = tf_train_masked(tfm, tr_ids, tr_mask, args.budget,
                                    lr=args.tf_lr, log=log)
        t0 = time.time()
        ce_v = tf_eval_masked(tfm, va_ids, va_mask, args.tf_ctx)
        log("[QA-tf] FINAL val masked_ce=%.4f ppl=%.2f | steps=%d tps=%.0f eval=%.0fs"
            % (ce_v, math.exp(min(ce_v, 20)), steps,
               steps * 64 / dt, time.time() - t0))
    log("[DONE] qa_lm")
    logf.close()


if __name__ == "__main__":
    main()
