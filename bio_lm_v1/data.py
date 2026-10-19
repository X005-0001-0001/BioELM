import array
import os
import random
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_stream(path):
    a = array.array("i")
    with open(path, "rb") as f:
        a.fromfile(f, os.path.getsize(path) // 4)
    return list(a)


def story_streams():
    story = load_stream(os.path.join(ROOT, "story_v63.txt.len16.norm.ids.bin"))
    warm = load_stream(os.path.join(HERE, "..", "warm_article.norm.ids.bin")) \
        if os.path.exists(os.path.join(HERE, "..", "warm_article.norm.ids.bin")) else []
    if not warm:
        p = os.path.join(ROOT, "_offtask", "bio_llm_side", "bio_llm", "warm_article.norm.ids.bin")
        warm = load_stream(p) if os.path.exists(p) else []
    return story, warm


def pick_word_bank(tok_path, max_words=120):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tok_path)
    vocab = tk.get_vocab()
    cjk = re.compile(r"^[\u4e00-\u9fff]{1,2}$")
    words = []
    for tstr, tid in vocab.items():
        s = tk.decode([tid], skip_special_tokens=False)
        s = unicodedata.normalize("NFKC", s)
        if cjk.match(s):
            words.append((tid, s))
    words.sort(key=lambda x: x[1])
    return [w for w, _ in words[:max_words]]


def gen_synth_corpus(out_path, tok_path, n_tokens=200000, seed=11, n_class=4):
    words = pick_word_bank(tok_path)
    if len(words) < 40:
        raise RuntimeError("word bank too small: %d" % len(words))
    rng = random.Random(seed)
    cls = [words[i::n_class] for i in range(n_class)]
    punct = []
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tok_path)
    for tstr, tid in tk.get_vocab().items():
        s = tk.decode([tid], skip_special_tokens=False)
        if s in ("。", "，", "？", "：", "、", "！"):
            punct.append(tid)
    if not punct:
        punct = [words[-1]]
    templates = [
        ["A", "B", "C", "P"],
        ["D", "A", "B", "P"],
        ["A", "C", "P", "A", "B", "P"],
        ["B", "C", "D", "P"],
        ["A", "A", "B", "P"],
        ["D", "D", "A", "C", "P"],
        ["B", "P", "A", "C", "P"],
        ["A", "C", "B", "P", "D", "A", "P"],
    ]
    slot_pool = {"A": cls[0], "B": cls[1], "C": cls[2], "D": cls[3], "P": punct}
    ids = []
    while len(ids) < n_tokens:
        tpl = templates[rng.randrange(len(templates))]
        for slot in tpl:
            pool = slot_pool[slot]
            if pool is punct:
                ids.append(pool[rng.randrange(len(pool))])
            else:
                r = rng.random()
                idx = int((r ** 2.2) * len(pool))
                ids.append(pool[min(idx, len(pool) - 1)])
    with open(out_path, "wb") as f:
        array.array("i", ids).tofile(f)
    return len(ids)


def make_windows(ids, k, val_tail=4000, val_rand=800, seed=7):
    rng = random.Random(seed)
    n = len(ids)
    idx = list(range(k, n))
    tail = idx[-val_tail:]
    rest = idx[:-val_tail]
    randv = rng.sample(rest, min(val_rand, len(rest)))
    randv_set = set(randv)
    train = [i for i in rest if i not in randv_set]
    tailset = set(tail)

    def build(positions):
        ctx = []
        tgt = []
        for p in positions:
            ctx.append(ids[p - k:p])
            tgt.append(ids[p])
        return ctx, tgt

    return build(train), build(sorted(randv)), build(tail)


if __name__ == "__main__":
    out = os.path.join(HERE, "synth.ids.bin")
    tok = os.path.join(ROOT, "tokenizer_v63b4000_tokenizer.json")
    n = gen_synth_corpus(out, tok)
    print("[SYNTH] %d tokens -> %s" % (n, out))
