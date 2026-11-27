# Changelog

## 2026-11-27
- Device-aware training: automatic CUDA / MPS / CPU selection, `ASURA_DEVICE` override.
- Colab/Kaggle notebook `asura_gpu.ipynb`.
- Resume support for sweeps and U3 after out-of-memory crashes on the 8 GB laptop.
- U1 (trust-optimization Goodhart curve) built.
- Documentation site (MkDocs Material) and GitHub Pages workflow.
- `INTEGRATIONS.md`: ~70 tools and APIs by category.

## 2026-11-26
- Asura v0: transformer (RMSNorm, RoPE, SwiGLU, tied embeddings), byte tokenizer, synthetic tasks,
  training loop, evals, seeds × CI sweep runner.
- Smoke test: 95% exact-match on unseen prompts after 9 CPU-minutes.
- Imported Algo lending simulation: results for 5 hypotheses.
- Clean Data Paradox messy-data knobs and sweep.
- U3 self-explanation faithfulness experiment.
