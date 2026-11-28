# Contributing

## Layout
```text
asura/                 core library: model, tokenizer, data, train, eval, device
experiments/
  configs/             training configs and sweep files
  run.py               conditions × seeds runner
  imported_algo/       lending simulation
  faithfulness/        U3
  trust_opt/           U1
papers/
  TEMPLATE.md          claim → experiment worksheet
  <paper>/PLAN.md      plan, results and verdict per paper (rendered in the docs)
docs/                  this site (MkDocs Material)
asura_gpu.ipynb        Colab/Kaggle notebook
```

## Conventions
- **New behaviour goes behind a flag that defaults off**, so baselines never change silently.
- **Every experiment gets a PLAN.md** with hypothesis, null result, setup, results and verdict.
- **Results must include seeds and CIs.** No single-run claims.
- Keep files readable top to bottom. This is a learning codebase.

## Docs
Write in `docs/`. Research pages embed `papers/*/PLAN.md`, so update the plan and the site follows.
API pages are generated from docstrings.

```powershell
.venv\Scripts\python -m mkdocs serve
```

## Tests
Run the test suite with:

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

GitHub Actions runs this suite for pull requests and pushes to `main`.

## Not committed
`runs/`, checkpoints, `.venv/`, `dist/`, the built `site/`, and the source paper PDFs/text
(`papers/source/`, `papers/text/`).
