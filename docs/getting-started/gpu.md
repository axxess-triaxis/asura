# Run on a free GPU

Asura automatically uses the best device it finds: NVIDIA GPU → Apple GPU → CPU
(`asura/device.py`). No code changes needed.

Force a device with an environment variable:

```powershell
$env:ASURA_DEVICE = "cpu"   # or cuda, mps
```

## Option A: Google Colab (easiest)

1. Open [colab.research.google.com](https://colab.research.google.com) → *File → Upload notebook* → `asura_gpu.ipynb`.
2. *Runtime → Change runtime type → T4 GPU → Save*.
3. Run the cells. Cell 2 clones the repo or asks you to upload `asura.zip`.
4. The last cell downloads `asura_runs.zip` with all results.

Free tier: sessions end after a few hours of inactivity, so download results promptly.

## Option B: Kaggle Notebooks (more free hours)

1. Verify your phone number in Kaggle settings. GPU is unavailable until you do.
2. *Create → New Notebook → File → Import notebook* → `asura_gpu.ipynb`.
3. Right panel: *Accelerator → GPU T4 x2*.
4. *Add Input → Upload* → `asura.zip` as a dataset.
5. *Run all*. Results appear in the *Output* tab.

Free tier: ~30 GPU-hours per week.

## Building `asura.zip`

From the project root:

```powershell
.venv\Scripts\python -c "import shutil; shutil.make_archive('dist/asura', 'zip', '.', None)"
```

Or clone directly from GitHub inside the notebook (no zip needed):

```python
!git clone https://github.com/<your-username>/asura.git && cd asura
```

## What a GPU unlocks

| | Laptop CPU | T4 GPU |
|---|---|---|
| Smoke test (0.36M params, 1.5k steps) | ~9 min | ~1-2 min |
| Seeds per condition (practical) | 3 | 5-10 |
| Largest practical model | ~1M params | ~10-50M params |
| Several experiments at once | ✗ (8 GB RAM OOM) | ✓ |

Scale a model from the command line:

```bash
python -m asura.train experiments/configs/smoke.json \
  --set name=smoke_10M model.d_model=384 model.n_layers=6 model.n_heads=6 train.steps=3000
```
