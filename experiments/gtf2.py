import torch
import torch.nn as nn
import torch.nn.functional as F


class Block2(nn.Module):
    def __init__(self, d, heads, ff, dropout=0.1):
        super().__init__()
        self.drop = dropout
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + F.dropout(a, self.drop, self.training)
        h = self.ln2(x)
        return x + F.dropout(self.mlp(h), self.drop, self.training)


class TinyGPT2(nn.Module):
    def __init__(self, vocab, k=128, d=64, layers=1, heads=4, ff=128, dropout=0.1):
        super().__init__()
        self.k = k
        self.drop = dropout
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(k, d)
        self.blocks = nn.ModuleList([Block2(d, heads, ff, dropout) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ctx):
        b, k = ctx.shape
        x = self.emb(ctx) + self.pos(torch.arange(k, device=ctx.device)).unsqueeze(0)
        x = F.dropout(x, self.drop, self.training)
        mask = torch.triu(torch.ones(k, k, dtype=torch.bool, device=ctx.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        h = self.ln_f(x)[:, -1, :]
        return h @ self.head.weight.t()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
