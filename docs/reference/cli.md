# Command-line reference

All commands run from the project root with the virtual environment's Python
(`.venv\Scripts\python` on Windows, `.venv/bin/python` elsewhere).

## `python -m asura.train CONFIG [--seed N] [--set KEY=VALUE ...]`
Train one model.

| Argument | Meaning |
|---|---|
| `CONFIG` | Path to a training config JSON |
| `--seed` | Random seed (default 0) |
| `--set` | Override config keys with dotted paths, e.g. `model.n_layers=4 train.lr=3e-4` |

Writes `runs/<name>/seed<N>/{config.json, log.jsonl, final.json, ckpt.pt}`.

## `python experiments/run.py SWEEP`
Run every condition × seed in a sweep file and print mean ± 95% CI. Resumable.
Writes `runs/<sweep name>/summary.json`.

## `python experiments/imported_algo/sim.py [--seeds N]`
Imported Algo lending simulation. Default 5 seeds. Writes `runs/imported_algo/results.json`.

## `python experiments/faithfulness/u3.py [options]`
| Option | Default | Meaning |
|---|---|---|
| `--seeds` | 3 | Seeds per condition |
| `--steps` | 1500 | Training steps per run |
| `--orders` | `answer_first,explain_first` | Explanation placement |
| `--rhos` | `0.75,0.95` | How often the shortcut agrees with the rule in training |
| `--threads` | all | CPU threads |

Resumable per condition. Writes `runs/faithfulness_u3/results.json`.

## `python experiments/trust_opt/u1.py [options]`
| Option | Default | Meaning |
|---|---|---|
| `--seeds` | 2 | Seeds |
| `--lams` | `0,0.25,0.5,0.75` | Weight on rater approval vs. correctness |
| `--sft_steps` | 1500 | Base-model training steps |
| `--rounds` | 4 | Best-of-n optimization rounds |
| `--per_round` | 256 | Prompts per round |
| `--n` | 8 | Samples per prompt |
| `--threads` | all | CPU threads |

Writes `runs/trust_opt_u1/results.json`.

## Docs
```powershell
.venv\Scripts\python -m mkdocs serve          # live preview at http://127.0.0.1:8000
.venv\Scripts\python -m mkdocs build --strict # build ./site, fail on any broken link
```

## Environment variables
| Variable | Effect |
|---|---|
| `ASURA_DEVICE` | Force `cpu`, `cuda` or `mps` |
| `PYTHONIOENCODING=utf-8` | Correct printing of `±` and Assamese digits on Windows |
