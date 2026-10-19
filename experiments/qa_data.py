import argparse
import array
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

from tokenizers import Tokenizer


def norm_hash(s):
    return hashlib.sha1(s.strip().encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "dsr1_v5_0001.jsonl"))
    ap.add_argument("--tok", default=os.path.join(ROOT, "tokenizer_v63b4000_tokenizer.json"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "datasets"))
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tk = Tokenizer.from_file(args.tok)
    v = tk.get_vocab()
    T = {t: i for t, i in v.items()}
    ids = {k: v[k] for k in ("<|bos|>", "<|user|>", "<|assistant|>", "<|end|>", "<|eos|>")}
    print("[QA-DATA] vocab=%d specials=%s" % (len(v), ids))

    recs = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))

    seen = set()
    uniq = []
    n_dup = 0
    for r in recs:
        h = norm_hash(json.dumps(r, ensure_ascii=False, sort_keys=True))
        if h in seen:
            n_dup += 1
            continue
        seen.add(h)
        uniq.append(r)
    print("[QA-DATA] records=%d dedup_dropped=%d unique=%d" % (len(recs), n_dup, len(uniq)))

    rng = random.Random(args.seed)
    order = list(range(len(uniq)))
    rng.shuffle(order)
    n_val = max(1, int(len(uniq) * args.val_frac))
    val_idx = set(order[:n_val])
    train_recs = [uniq[i] for i in order[n_val:]]
    val_recs = [uniq[i] for i in order[:n_val]]

    tr_user = set()
    tr_asst = set()
    for r in train_recs:
        for m in r["messages"]:
            h = norm_hash(m["content"])
            (tr_user if m["role"] == "user" else tr_asst).add(h)
    leak_u = 0
    leak_a = 0
    keep_val = []
    for r in val_recs:
        bad = False
        for m in r["messages"]:
            h = norm_hash(m["content"])
            if m["role"] == "user" and h in tr_user:
                leak_u += 1
                bad = True
            if m["role"] == "assistant" and h in tr_asst:
                leak_a += 1
                bad = True
        if not bad:
            keep_val.append(r)
    print("[QA-DATA] leak check: val with user-overlap=%d asst-overlap=%d kept=%d"
          % (leak_u, leak_a, len(keep_val)))
    tr_user_blob = "\n".join(m["content"].strip() for r in train_recs
                             for m in r["messages"] if m["role"] == "user")
    tr_asst_blob = "\n".join(m["content"].strip() for r in train_recs
                             for m in r["messages"] if m["role"] == "assistant")
    sub_u = 0
    sub_a = 0
    for r in keep_val:
        for m in r["messages"]:
            probe = m["content"].strip()[:80]
            if not probe:
                continue
            if m["role"] == "user" and probe in tr_user_blob:
                sub_u += 1
            if m["role"] == "assistant" and probe in tr_asst_blob:
                sub_a += 1
    print("[QA-DATA] near-dup check: val 80-char probes found in train: user=%d "
          "assistant=%d (inspect: boilerplate vs content)" % (sub_u, sub_a))

    def build(records):
        all_ids = []
        all_mask = []
        for r in records:
            seq = [ids["<|bos|>"]]
            msk = [0]
            for m in r["messages"]:
                if m["role"] == "user":
                    seq.append(ids["<|user|>"])
                    msk.append(0)
                    e = tk.encode(m["content"]).ids
                    seq.extend(e)
                    msk.extend([0] * len(e))
                    seq.append(ids["<|end|>"])
                    msk.append(0)
                else:
                    seq.append(ids["<|assistant|>"])
                    msk.append(0)
                    e = tk.encode(m["content"]).ids
                    seq.extend(e)
                    msk.extend([1] * len(e))
                    seq.append(ids["<|end|>"])
                    msk.append(1)
            seq.append(ids["<|eos|>"])
            msk.append(1)
            all_ids.extend(seq)
            all_mask.extend(msk)
        return all_ids, all_mask

    tr_ids, tr_mask = build(train_recs)
    va_ids, va_mask = build(keep_val)
    max_id = max(max(tr_ids), max(va_ids))
    print("[QA-DATA] train tokens=%d (masked-learn=%d, %.1f%%) | val tokens=%d "
          "(learn=%d, %.1f%%) | max_id=%d"
          % (len(tr_ids), sum(tr_mask), 100.0 * sum(tr_mask) / len(tr_mask),
             len(va_ids), sum(va_mask), 100.0 * sum(va_mask) / len(va_mask), max_id))

    def save(name, toks, mask):
        with open(os.path.join(args.outdir, name + ".ids.bin"), "wb") as f:
            array.array("i", toks).tofile(f)
        with open(os.path.join(args.outdir, name + ".mask.bin"), "wb") as f:
            array.array("B", mask).tofile(f)
        print("[QA-DATA] saved %s: ids=%d" % (name, len(toks)))

    save("qa_train", tr_ids, tr_mask)
    save("qa_val", va_ids, va_mask)
    with open(os.path.join(args.outdir, "qa_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": len(v), "max_id": max_id,
                   "train_records": len(train_recs), "val_records": len(keep_val),
                   "train_tokens": len(tr_ids), "val_tokens": len(va_ids),
                   "train_learn_tokens": sum(tr_mask),
                   "val_learn_tokens": sum(va_mask)}, f, ensure_ascii=False, indent=1)
    print("[QA-DATA] done")


if __name__ == "__main__":
    main()
