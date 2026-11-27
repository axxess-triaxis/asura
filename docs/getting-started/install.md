# Install & first run

## Requirements
- Python 3.10+ (developed on 3.14)
- ~1 GB free RAM for default experiments, more if you run several at once
- No GPU needed. See [Run on a free GPU](gpu.md) to go faster.

## Install

=== "Windows (PowerShell)"

    ```powershell
    git clone https://github.com/<your-username>/asura.git
    cd asura
    python -m venv .venv
    .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv\Scripts\python -m pip install numpy pypdf
    ```

=== "macOS / Linux"

    ```bash
    git clone https://github.com/<your-username>/asura.git
    cd asura
    python -m venv .venv
    .venv/bin/pip install torch numpy pypdf
    ```

!!! tip "CPU-only PyTorch"
    The `--index-url .../whl/cpu` wheel is much smaller than the default CUDA build. Use it on
    machines without an NVIDIA GPU.

## Your first model

```powershell
.venv\Scripts\python -m asura.train experiments\configs\smoke.json
```

This trains a 0.36M-parameter Asura on two synthetic tasks (2-digit addition and string
reversal) for 1,500 steps. About 9 minutes on a laptop CPU, 1-2 minutes on a T4 GPU.

What you should see:

```text
[asura] smoke seed=0 params=0.36M device=cpu -> runs\smoke\seed0
  step=0 loss=5.52 ... acc_all=0
  step=500 loss=0.2994 ... acc_reverse=0.9806 acc_add=0.03093
  step=1499 loss=0.02625 ... acc_reverse=1 acc_add=0.8557 acc_all=0.93
```

Note the order: **reversal is learned first** (a copying pattern attention handles easily),
and **addition much later** (it needs carries). Watching this is a lesson in itself.

## What got written

```text
runs/smoke/seed0/
├── config.json   # exact config + seed, for reproducibility
├── log.jsonl     # one JSON line per logged step (loss, lr, grad_norm, evals)
├── final.json    # final metrics on the full held-out test set
└── ckpt.pt       # model weights + config
```

## Your first experiment

```powershell
.venv\Scripts\python experiments\run.py experiments\configs\smoke_sweep.json
```

This trains 1-layer vs. 3-layer models with 3 seeds each and prints mean ± 95% CI.
See [Configs & sweeps](../guides/experiments.md).
