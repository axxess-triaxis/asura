# Configs & sweeps

## Training config
One JSON file describes one model + dataset + training run:

```json
{
  "name": "smoke",
  "data":  {"type": "synthetic", "tasks": ["add", "reverse"], "digits": 2, "n_train": 8000, "n_test": 400},
  "model": {"context_len": 32, "d_model": 96, "n_layers": 3, "n_heads": 4},
  "train": {"steps": 1500, "batch_size": 64, "lr": 2e-3, "warmup": 100, "weight_decay": 0.1,
            "log_every": 100, "eval_every": 500},
  "eval":  {"task_samples": 200, "loss_batches": 5}
}
```

Run it, optionally overriding any key with dotted paths:

```powershell
.venv\Scripts\python -m asura.train experiments\configs\smoke.json --seed 1 --set model.n_layers=4 train.lr=1e-3
```

Values in `--set` are parsed as JSON (`4` → int, `[1,2]` → list, `true` → bool), otherwise kept as strings.

## Sweep file
A sweep runs several **conditions** × several **seeds** on top of a base config.

=== "Grid (every combination)"

    ```json
    {
      "base": "experiments/configs/smoke.json",
      "name": "depth_vs_width",
      "seeds": [0, 1, 2],
      "grid": {"model.n_layers": [2, 4], "model.d_model": [64, 128]},
      "metric": "acc_all"
    }
    ```

=== "Named conditions"

    ```json
    {
      "base": "experiments/configs/clean_data_base.json",
      "name": "clean_data_paradox",
      "seeds": [0, 1, 2],
      "conditions": {
        "C0_clean_only":   {"data.messy_frac": 0.0},
        "C1_messy_inputs": {"data.messy_frac": 0.5}
      },
      "metric": "acc_all",
      "report": ["acc_add", "acc_add~messy~odd"]
    }
    ```

```powershell
.venv\Scripts\python experiments\run.py experiments\configs\clean_data_paradox.json
```

Output:

```text
C0_clean_only        acc_all=0.2800 +- 0.1256  (n=3, params=0.36M)
    acc_add                      0.5638 +- 0.2528
    acc_add~messy~odd            0.0000 +- 0.0000
```

- `metric`: the primary metric, shown on the condition line.
- `report`: extra metrics printed beneath it.
- Every numeric metric's mean and CI is saved in `runs/<name>/summary.json` under `all_metrics`.

## Resuming
If a sweep crashes (e.g. out of memory), run the same command again. Any seed with a
`final.json` is loaded instead of retrained.

## Standalone experiment scripts
Some experiments need custom loops and live in their own folders:

| Script | Paper | Notes |
|---|---|---|
| `experiments/imported_algo/sim.py --seeds 5` | Imported Algo | Logistic-regression lending sim, ~1 min |
| `experiments/faithfulness/u3.py --seeds 3` | Uncanny Valley | Resumable per condition |
| `experiments/trust_opt/u1.py --seeds 2` | Hardest Tradeoff | SFT then best-of-n optimization |

All three accept `--threads N` to limit CPU use.
