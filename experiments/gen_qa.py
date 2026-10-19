import argparse
import json
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
torch.set_num_threads(min(16, os.cpu_count() or 1))
DEV = torch.device(os.environ.get("BIO_DEV", "cpu"))

from tokenizers import Tokenizer
from biolssm import BioGSSM


@torch.no_grad()
def generate_answer(rnn, tk, question, bos, user, assistant, end, eos,
                    n_new=400, temp=0.7, top_p=0.9, seed=0):
    g = torch.Generator().manual_seed(seed)
    st = rnn.init_stream_state(1)
    prompt = [bos, user] + tk.encode(question).ids + [end, assistant]
    logits = None
    for t in prompt:
        logits = rnn.step_batch(torch.tensor([t], device=DEV),
                                torch.tensor([0], device=DEV), st, train=False)
    out = []
    for _ in range(n_new):
        z = logits[0].clone()
        z[2] = -1e9
        z[3] = -1e9
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
        if nxt in (end, eos):
            break
        out.append(nxt)
        logits = rnn.step_batch(torch.tensor([nxt], device=DEV),
                                torch.tensor([0], device=DEV), st, train=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "dsr1_v5_0001.jsonl"))
    ap.add_argument("--tok", default=os.path.join(ROOT, "tokenizer_v63b4000_tokenizer.json"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt_gssm_qa_gpu.pt"))
    ap.add_argument("--h", type=int, default=384)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--log", default=os.path.join(HERE, "logs", "qa_samples.log"))
    args = ap.parse_args()

    logf = open(args.log, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    tk = Tokenizer.from_file(args.tok)
    recs = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    rng = __import__("random").Random(31)
    idxs = rng.sample(range(len(recs)), args.n)

    torch.manual_seed(43)
    rnn = BioGSSM(2754, d=args.d, h=args.h, m=args.m, seed=43)
    ck = torch.load(args.ckpt, map_location="cpu")
    for k, v in ck.items():
        if k == "E":
            rnn.E.copy_(v.to(DEV))
            continue
        rnn.params()[k].copy_(v.to(DEV))

    log("=" * 78)
    log("[GEN] ckpt=%s dev=%s temp=%.2f" % (os.path.basename(args.ckpt), DEV, args.temp))
    for i, ri in enumerate(idxs):
        msgs = recs[ri]["messages"]
        q = next(m["content"] for m in msgs if m["role"] == "user")
        gold = next(m["content"] for m in msgs if m["role"] == "assistant")
        out = generate_answer(rnn, tk, q, 0, 4, 5, 6, 1,
                              temp=args.temp, seed=100 + i)
        ans = tk.decode(out, skip_special_tokens=True)
        log("--- sample %d (record %d) ---" % (i + 1, ri))
        log("[Q] " + q[:160].replace("\n", " "))
        log("[BIO] " + ans[:400].replace("\n", " "))
        log("[GOLD] " + gold[:200].replace("\n", " ") + " ...")
    log("[DONE] gen_qa")
    logf.close()


if __name__ == "__main__":
    main()
