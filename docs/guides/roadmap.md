# Learning roadmap

A 10-week path from "what is attention?" to running your own research, all inside Asura.

| Stage | Goal | Do this in Asura | Read |
|---|---|---|---|
| **1. Foundations** (wk 1-2) | Understand every line of `model.py` | Re-derive attention by hand. Predict the parameter count before changing `d_model`/`n_layers` ([exercise](../concepts/architecture.md#counting-parameters)) | Karpathy, *Let's build GPT* (video); Vaswani et al., *Attention Is All You Need* |
| **2. Training dynamics** (wk 3-4) | Read loss curves | Sweep LR 1e-4 → 1e-2 and find where training diverges. Remove warmup. Plot `log.jsonl` | nanoGPT source; Google's *Deep Learning Tuning Playbook* |
| **3. Scale & data** (wk 5-6) | Scaling laws at toy scale | Train 5 sizes at a fixed token budget, fit loss vs. params. Add a BPE tokenizer | Kaplan et al. 2020; Hoffmann et al. 2022 (Chinchilla) |
| **4. Evals** (wk 7-8) | Measure the right thing | Build a length-generalization eval. Report CIs, never single runs | Anthropic, *A statistical approach to model evaluations*; HELM |
| **5. Interpretability** (wk 9-10) | Look inside | Extend U3: attention patterns, find induction heads, activation patching | Anthropic, *Transformer Circuits* thread: start with *A Mathematical Framework for Transformer Circuits* |
| **6. Your research** (ongoing) | Empirical tests of your papers | One claim at a time with the [worksheet](new-experiment.md). ≥5 seeds before claiming anything | - |

## Research hygiene (what frontier labs look for)
- One variable changes per comparison. Everything else stays fixed and gets logged.
- Several seeds and confidence intervals. One run is an anecdote.
- Held-out evaluation, so you measure generalization, not memorization.
- Honest negative and null results. They build more trust than cherry-picked wins.
- Short public write-ups: hypothesis → setup → plot → verdict → caveats.
