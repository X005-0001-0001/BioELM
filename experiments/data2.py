import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "bio_lm_v1"))

from data import pick_word_bank


def bank():
    return pick_word_bank(os.path.join(ROOT, "tokenizer_v63b4000_tokenizer.json"),
                          max_words=120)


def gen_longrange_gapmix(n_sent=8000, seed=13, bank_words=None,
                         bands=((5, 15, 0.3), (16, 40, 0.4), (41, 70, 0.3))):
    b = bank_words or bank()
    markers = b[0:4]
    verbs = b[4:8]
    sep = b[8]
    period = b[9]
    filler = b[10:110]
    rng = random.Random(seed)
    ids = []
    crit = []
    for _ in range(n_sent):
        m = rng.randrange(4)
        r = rng.random()
        acc = 0.0
        lo, hi = bands[-1][0], bands[-1][1]
        for blo, bhi, frac in bands:
            acc += frac
            if r <= acc:
                lo, hi = blo, bhi
                break
        ids.append(markers[m])
        crit.append(0)
        nf = rng.randint(lo, hi)
        for _ in range(nf):
            ids.append(filler[rng.randrange(len(filler))])
            crit.append(0)
        ids.append(sep)
        crit.append(1)
        ids.append(verbs[m])
        crit.append(0)
        ids.append(period)
        crit.append(0)
    return ids, crit


def gen_longrange(n_sent=8000, filler_lo=30, filler_hi=70, seed=13, bank_words=None):
    b = bank_words or bank()
    markers = b[0:4]
    verbs = b[4:8]
    sep = b[8]
    period = b[9]
    filler = b[10:110]
    rng = random.Random(seed)
    ids = []
    crit = []
    for _ in range(n_sent):
        m = rng.randrange(4)
        ids.append(markers[m])
        crit.append(0)
        nf = rng.randint(filler_lo, filler_hi)
        for _ in range(nf):
            ids.append(filler[rng.randrange(len(filler))])
            crit.append(0)
        ids.append(sep)
        crit.append(1)
        ids.append(verbs[m])
        crit.append(0)
        ids.append(period)
        crit.append(0)
    return ids, crit


def gen_recall(n_tokens=200000, pool=32, delay=20, seed=17, bank_words=None):
    b = bank_words or bank()
    pool_ids = b[10:10 + pool]
    rng = random.Random(seed)
    ids = [pool_ids[rng.randrange(pool)] for _ in range(n_tokens)]
    tgt = []
    crit = []
    for t in range(n_tokens):
        if t >= delay:
            tgt.append(ids[t - delay])
            crit.append(1)
        else:
            tgt.append(ids[t])
            crit.append(0)
    return ids, tgt, crit


def shift_next(ids):
    return list(ids[1:]) + [ids[0]]


def to_streams(ids, tgt, crit, n_streams, T=None):
    import torch
    n = len(ids)
    T = T or n // n_streams
    usable = T * n_streams
    cut = n - usable
    ids = ids[cut:]
    tgt = tgt[cut:]
    crit = crit[cut:]
    st = torch.tensor(ids, dtype=torch.long).reshape(n_streams, T)
    tg = torch.tensor(tgt, dtype=torch.long).reshape(n_streams, T)
    cr = torch.tensor(crit, dtype=torch.bool).reshape(n_streams, T)
    return st, tg, cr


def sliding_windows(ids, tgt, crit, ctx, stride=1):
    import torch
    n = len(ids)
    starts = list(range(0, n - 1, stride))
    xs = torch.zeros(len(starts), ctx, dtype=torch.long)
    ys = torch.zeros(len(starts), dtype=torch.long)
    cs = torch.zeros(len(starts), dtype=torch.bool)
    for i, p in enumerate(starts):
        lo = max(0, p + 1 - ctx)
        seq = ids[lo:p + 1]
        xs[i, ctx - len(seq):] = torch.tensor(seq, dtype=torch.long)
        ys[i] = tgt[p]
        cs[i] = bool(crit[p])
    return xs, ys, cs
