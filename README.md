# Asura

A small language model you build, train and evaluate yourself. It's also a
test bench for putting research claims through controlled experiments.

Written for a CPU laptop (no NVIDIA GPU, 8 GB RAM). Every file is short enough
to read in one sitting.

```
asura/model.py      the transformer (RMSNorm, RoPE, causal attention, SwiGLU, tied embeddings)
asura/tokenizer.py  byte-level tokenizer (vocab 256)
asura/data.py       text datasets + synthetic tasks with ground truth
asura/train.py      training loop (AdamW, warmup + cosine LR, grad clip, JSONL logs)
asura/eval.py       held-out loss/perplexity + exact-match accuracy on unseen prompts
experiments/run.py  grid of conditions x seeds -> mean, std, 95% CI
papers/TEMPLATE.md  worksheet that turns a paper claim into an experiment
```

## Quick start (PowerShell)

```powershell
.venv\Scripts\python -m asura.train experiments\configs\smoke.json
.venv\Scripts\python experiments\run.py experiments\configs\smoke_sweep.json
```

To train on your own text, use a data block such as `{"type": "text", "path": "data/my_corpus"}`
(a folder of .txt files).

## Learning roadmap

| Stage | Goal | Do this in Asura | Read |
|---|---|---|---|
| 1. Foundations (wk 1-2) | Understand every line of `model.py` | Re-derive attention by hand. Change `n_heads` and `d_model` and predict the param count before you run it. | Karpathy, "Let's build GPT" (video); *Attention Is All You Need* |
| 2. Training dynamics (wk 3-4) | Read loss curves | Sweep the LR (1e-4 to 1e-2) and see where training diverges. Turn warmup off. Plot `log.jsonl`. | nanoGPT source; *Deep Learning Tuning Playbook* (Google) |
| 3. Scale and data (wk 5-6) | Scaling laws at toy scale | Train 5 model sizes at a fixed token budget and fit loss vs. params. Add a BPE tokenizer. | Kaplan et al. 2020; Hoffmann et al. 2022 (Chinchilla) |
| 4. Evals (wk 7-8) | Measure the right thing | Add a length-generalization eval: train on short inputs, test on long ones. Report CIs, not single runs. | Anthropic, "Statistical approach to model evals"; HELM |
| 5. Interpretability (wk 9-10) | Look inside | Hook the attention patterns, find induction heads, try ablations and linear probes. | Anthropic, *Transformer Circuits* thread (start with "A Mathematical Framework") |
| 6. Your papers (ongoing) | Empirical tests | One claim at a time with `papers/TEMPLATE.md`. Get ≥5 seeds before you claim anything. | - |

When you outgrow CPU, rent a single GPU by the hour (an A10 or L4 is plenty for
models up to ~100M params). The same code runs there if you add `.to("cuda")`.

## Research hygiene (what labs look for)
- One variable changes per comparison. Everything else stays fixed and gets logged.
- Run several seeds and report confidence intervals. One run is an anecdote.
- Evaluate on held-out prompts so you're measuring generalization, not memorization.
- Report negative and null results honestly. That builds more trust than a cherry-picked win.
- Write each experiment up as a short public note: hypothesis, setup, plot, verdict, caveats.
