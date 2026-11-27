# Training

`asura/train.py` is a plain PyTorch loop. There's no framework, so every step is visible.

## The loop

```python
for step in range(steps):
    set lr from schedule
    x, y, mask = data.batch("train", batch_size, context_len)
    _, loss = model(x, y, mask)             # cross-entropy on answer tokens
    loss.backward()
    clip_grad_norm_(params, grad_clip)      # stability
    optimizer.step()
    every log_every:  write loss, lr, grad_norm to log.jsonl
    every eval_every: run evals, write to log.jsonl
```

## Optimizer: AdamW
- β = (0.9, 0.95), standard for transformers.
- **Weight decay only on matrices** (dim ≥ 2). Norm gains are excluded, because decaying them hurts.

## Learning-rate schedule
Linear **warmup** for `warmup` steps, then **cosine decay** to `min_lr_frac × lr` (default 10%):

$$\eta_t = \begin{cases} \eta_\max \frac{t+1}{T_w} & t < T_w \\ \eta_\min + \tfrac12(\eta_\max-\eta_\min)\left(1+\cos\pi\frac{t-T_w}{T-T_w}\right) & \text{otherwise}\end{cases}$$

Warmup keeps Adam's early, noisy updates from blowing up the network. Cosine decay lets it
settle into a good minimum at the end.

## Config reference (`train` block)

| Key | Default | Meaning |
|---|---|---|
| `steps` | required | Optimizer steps |
| `batch_size` | required | Sequences per step |
| `lr` | required | Peak learning rate |
| `warmup` | required | Warmup steps |
| `min_lr_frac` | 0.1 | Final LR as a fraction of the peak |
| `weight_decay` | 0.1 | AdamW decay on matrices |
| `betas` | [0.9, 0.95] | Adam betas |
| `grad_clip` | 1.0 | Max global grad norm |
| `log_every` | 50 | Steps between log lines |
| `eval_every` | 500 | Steps between evals |
| `threads` | all cores | CPU threads (lower it to run beside other work) |

## Reading `log.jsonl`

```python
import json, matplotlib.pyplot as plt
rows = [json.loads(l) for l in open("runs/smoke/seed0/log.jsonl")]
plt.plot([r["step"] for r in rows if "loss" in r], [r["loss"] for r in rows if "loss" in r])
plt.yscale("log"); plt.xlabel("step"); plt.ylabel("train loss"); plt.show()
```

What to look for:

- **Loss spikes** together with a jump in `grad_norm`: the LR is too high.
- **Plateau, then a sudden drop**: a new skill clicked (common on algorithmic tasks).
- **Train loss falling while `val_loss` rises**: overfitting/memorization. Add data or decay.

## Reproducibility
Each run is seeded (`torch.manual_seed`, a dedicated `torch.Generator` for batches), and the exact
config and seed are written to `config.json`. Same seed + same config + same device gives the same
run. CPU and GPU can differ slightly in floating point.
