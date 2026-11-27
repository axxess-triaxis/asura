"""Asura: a small decoder-only transformer.

Modern-but-minimal architecture (same building blocks as Llama-style models):
  - token embedding (tied with the output head)
  - N x [RMSNorm -> causal self-attention w/ RoPE -> residual
         RMSNorm -> SwiGLU MLP -> residual]
  - final RMSNorm -> logits

Read this file top to bottom; every line is meant to be understandable.
"""
from dataclasses import dataclass, asdict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AsuraConfig:
    """Hyperparameters of an Asura model. Pass as `AsuraConfig(**cfg["model"])`."""
    vocab_size: int = 256        # byte-level tokenizer -> 256 symbols
    context_len: int = 256       # max sequence length
    d_model: int = 256           # width of the residual stream
    n_layers: int = 6
    n_heads: int = 8
    mlp_mult: float = 8 / 3      # SwiGLU hidden = mlp_mult * d_model
    dropout: float = 0.0
    rope_base: float = 10000.0

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(seq_len, head_dim, base, device):
    """Precompute cos/sin for rotary position embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D). Rotate pairs (x1, x2) by position-dependent angle.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: AsuraConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: AsuraConfig):
        super().__init__()
        hidden = int(cfg.mlp_mult * cfg.d_model)
        hidden = 64 * math.ceil(hidden / 64)
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.d_model), SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class Asura(nn.Module):
    """The Asura decoder-only transformer.

    Example:
        >>> model = Asura(AsuraConfig(d_model=96, n_layers=3, n_heads=4, context_len=32))
        >>> logits, loss = model(x, targets=y, loss_mask=m)   # training
        >>> out = model.generate(prompt_ids, 10, temperature=0)  # greedy decoding
    """

    def __init__(self, cfg: AsuraConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying
        cos, sin = rope_cache(cfg.context_len, cfg.d_model // cfg.n_heads, cfg.rope_base, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        # GPT-2 style: scale residual output projections by depth
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute next-token logits, and loss when `targets` are given.

        Args:
            idx: LongTensor (B, T) of token ids.
            targets: LongTensor (B, T) of next-token ids, or None.
            loss_mask: optional (B, T) 0/1 tensor; loss is averaged over positions where it is 1.

        Returns:
            (logits of shape (B, T, vocab), loss or None)
        """
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
        if loss_mask is not None:
            m = loss_mask.view(-1).float()
            loss = (loss * m).sum() / m.sum().clamp(min=1)
        else:
            loss = loss.mean()
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
        """Autoregressively extend `idx`. temperature=0 means greedy; top_k limits sampling;
        stops early once every sequence has emitted `stop_token`."""
        idx = idx.to(self.embed.weight.device)   # callers can pass CPU tensors on any device
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.context_len:])
            logits = logits[:, -1, :]
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_token is not None and (nxt == stop_token).all():
                break
        return idx
