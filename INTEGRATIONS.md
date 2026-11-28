# What Asura can integrate

Priority: ⭐ = do soon, high value for this project · ○ = useful later · Cost: free / $ / $$
"Paper" names which of your papers it helps test.

## 1. Compute (the biggest bottleneck: this laptop has no GPU and 8 GB RAM)
| Tool | What it gives you | Cost | Priority |
|---|---|---|---|
| Google Colab | Free T4 GPU in a notebook, runs Asura unchanged with `.to("cuda")` | free / $ (Pro) | ⭐ |
| Kaggle Notebooks | ~30 GPU-hours/week free, also hosts datasets | free | ⭐ |
| AWS (EC2 g5/g6 GPU, SageMaker training jobs, S3 for checkpoints) | Scalable, reproducible training. Your AWS MCP server is already connected | $$ | ○ |
| Modal / RunPod / Lambda / Vast.ai | Rent a single GPU by the hour, cheaper than big clouds | $ | ○ |
| Hugging Face Spaces (ZeroGPU) | Host a public demo of Asura | free / $ | ○ |

## 2. Experiment tracking & reproducibility
| Tool | Use | Cost | Priority |
|---|---|---|---|
| Weights & Biases | Live loss curves, sweep dashboards, shareable reports (good for applications) | free (personal) | ⭐ |
| TensorBoard | Local loss curves, zero signup | free | ○ |
| MLflow | Self-hosted tracking + model registry | free | ○ |
| Optuna | Hyperparameter search (fair tuning budget per condition) | free | ○ |
| Hydra | Cleaner config management than JSON + `--set` | free | ○ |
| Git + GitHub | Version control, public portfolio | free | ⭐ |
| GitHub Actions | Run a 30-second smoke test on every commit | free | ○ |
| DVC | Version datasets alongside code | free | ○ |
| pytest | Unit tests for model/data code | free | ⭐ |

## 3. Core ML libraries
| Tool | Use | Priority |
|---|---|---|
| PyTorch (installed) | Everything | - |
| `torch.compile`, mixed precision (on GPU) | 1.5-3× faster training | ○ |
| Hugging Face `transformers` | Compare Asura against pretrained small models (GPT-2, Pythia, SmolLM, Qwen-0.5B) | ⭐ |
| Hugging Face `datasets` | Load corpora/benchmarks in one line | ⭐ |
| `tokenizers` / `sentencepiece` / `tiktoken` / Karpathy's `minbpe` | Train a BPE tokenizer (roadmap Stage 3). SentencePiece handles Assamese script well | ⭐ |
| TRL (Hugging Face) | Proper RLHF / PPO / DPO for U1, instead of best-of-n | ⭐ (U1, U2) |
| PEFT / LoRA | Fine-tune bigger pretrained models cheaply | ○ |
| nanoGPT / nanochat (reference code) | Cross-check your implementation against a known-good one | ⭐ |

## 4. Datasets
**General pretraining (small scale)**
- TinyStories: perfect for CPU-size LMs that still produce coherent English ⭐
- FineWeb-Edu (samples), OpenWebText, Wikipedia dumps ○

**Indian / Assamese / Northeast (Clean Data Paradox, Imported Algo)**
- AI4Bharat: IndicCorp v2, Sangraha, IndicTrans2 parallel data, Aksharantar (transliteration) ⭐
- Assamese Wikipedia dump (`aswiki`) ⭐
- Bhashini (Government of India language platform): translation/ASR APIs, datasets ○
- Code-mixed (Hinglish/Assamese-English) social text corpora on HF Datasets ○

**Finance (Imported Algo)**
- RBI DBIE (Database on Indian Economy): aggregate credit data, free ⭐
- data.gov.in / MoSPI (ASUSE, the unincorporated enterprise survey): informal-sector statistics ⭐
- SIDBI / MSME Pulse reports: MSME credit trends ○
- Kaggle loan-default datasets (US/EU like LendingClub, Home Credit) to build a *real* "imported" model ⭐
- Account Aggregator framework (Sahamati): consented real financial data. Requires an NBFC/FIU partnership ○
- Credit bureaus (CIBIL, Experian, CRIF): restricted, partnership only ○

## 5. Evaluation frameworks & benchmarks
| Tool | Use | Paper |
|---|---|---|
| Inspect (UK AI Security Institute) | Modern eval framework used in safety research ⭐ | all |
| EleutherAI `lm-evaluation-harness` | Standard benchmarks on any HF model ⭐ | all |
| HELM (Stanford) | Holistic eval incl. fairness, calibration | Uncanny Valley |
| IndicGLUE / IndicXTREME / MILU / IndicGenBench | Indian-language benchmarks ⭐ | Clean Data |
| Anthropic model-written evals (GitHub: `anthropics/evals`) | Sycophancy datasets, a direct real-data version of U1 ⭐ | Uncanny Valley, Hardest Tradeoff |
| TruthfulQA, BBQ (bias), BOLD | Truthfulness and social bias | Uncanny Valley |
| HarmBench / JailbreakBench / StrongREJECT | Jailbreak measurement for the "hedging trains jailbreakers" claim | Hardest Tradeoff |
| Fairlearn / AIF360 | Group fairness metrics + mitigation (equal wrong-reject rates) ⭐ | Imported Algo |
| scipy / statsmodels / pingouin | Proper significance tests, bootstrap CIs | all |

## 6. Interpretability (U3 and beyond)
| Tool | Use | Priority |
|---|---|---|
| TransformerLens | Hook every activation, attention patterns, activation patching | ⭐ |
| nnsight | Interventions on large HF models | ○ |
| SAELens | Train/load sparse autoencoders (Anthropic-style features) | ⭐ |
| Neuronpedia | Browse SAE features of open models | ○ |
| pyvene | Causal interventions / interchange interventions | ○ |
| Captum | Attribution methods (integrated gradients) | ○ |
| circuitsvis | Visualize attention in notebooks | ○ |

## 7. LLM APIs (as judges, simulated users, data generators, comparison targets)
| API | Use in Asura | Priority |
|---|---|---|
| Anthropic Claude API | LLM-as-judge for hedging/softening (U1/U2), simulated users for the jailbreak-cascade sim, generating messy Assamese-English test cases. Test whether frontier models show sycophancy and unfaithful explanations | ⭐ |
| OpenAI API | Same, cross-provider comparison (are effects model-specific?) | ○ |
| Google Gemini API | Same | ○ |
| Amazon Bedrock | Many models behind one AWS API | ○ |
| Sarvam AI | Indian-built LLMs, compare against Western models on Indian financial contexts | ⭐ (Imported Algo) |
| AI4Bharat IndicTrans2 / Bhashini | Translate test sets into Assamese and other Indian languages | ⭐ (Clean Data) |
| Ollama / llama.cpp | Run open models (Llama, Qwen, Gemma) locally. Small ones fit in 8 GB | ○ |

## 8. Simulation & human studies
| Tool | Use | Paper |
|---|---|---|
| Mesa (agent-based modeling) | Jailbreak-cascade and trust-erosion simulations (Hardest Tradeoff §VIII) | Hardest Tradeoff |
| SimPy | Discrete-event sim of lending pipelines (fintech → NBFC → bank chain) | Imported Algo |
| Prolific / MTurk | Real human study: do hedged refusals make people re-prompt more? | Hardest Tradeoff |
| Google Forms / Qualtrics | Survey on perceived trust of hedged vs. blunt answers | Uncanny Valley |

## 9. Visualization & writing
- matplotlib / seaborn / plotly: figures for write-ups ⭐
- Jupyter / VS Code notebooks: exploratory analysis ⭐
- Overleaf (LaTeX), Zotero (citations), Semantic Scholar API / arXiv API (literature search) ○
- Publishing: arXiv (needs an endorser), LessWrong / Alignment Forum (where safety researchers read), personal blog, GitHub Pages ⭐

## Recommended next 5 integrations, in order
1. **Kaggle or Colab GPU.** Unlocks 10-100× bigger runs; no more OOM crashes.
2. **Weights & Biases.** Shareable dashboards for every experiment.
3. **Anthropic's sycophancy evals + Claude API.** Moves U1 from a toy model to real frontier models.
4. **TransformerLens + SAELens.** Turn U3 into a real interpretability study on a small pretrained model.
5. **AI4Bharat corpora + Fairlearn.** Real Assamese data for the Clean Data paper, and real fairness metrics for Imported Algo.

## Opt-in Asura integrations

Install the base project dependencies as usual, then install optional integrations only when needed:

```powershell
pip install -r requirements.txt -r requirements-integrations.txt
```

### Kaggle GPU training and public text data

Run Asura from a Kaggle notebook with a GPU accelerator enabled; `asura.device.get_device()` selects CUDA automatically. Store `KAGGLE_USERNAME` and `KAGGLE_KEY` in Kaggle Secrets and expose them to the notebook process through `kaggle_secrets.UserSecretsClient` (or configure the Kaggle CLI credentials in the environment). Never put credentials in the notebook or repository.

In a Kaggle notebook, read credentials from Kaggle Secrets into the process environment:

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
```

From the Asura repository root, download and train only from a named public dataset:

```python
!python -m asura.integrations.kaggle --dataset owner/public-text-dataset --config experiments/configs/smoke.json
```

The integration checks Kaggle's metadata and refuses datasets not confirmed public, safely extracts the archive, and accepts UTF-8 `.txt` files only. Convert other file formats to text yourself before training. The dataset identifier is required; no dataset is selected or downloaded implicitly.

### Local MLflow tracking

MLflow is disabled unless `tracking.enabled` is explicitly set to `true` in the training JSON config:

```json
{
  "tracking": {
    "enabled": true,
    "uri": "file:./mlruns",
    "experiment": "asura-local"
  }
}
```

Only local `file:` tracking URIs are accepted. Config parameters and final numeric metrics are logged; dataset paths, model checkpoints, prompts, and example text are not uploaded or logged as artifacts.

### LangSmith metrics-only tracing

LangSmith is disabled by default. To explicitly log a synthetic/public evaluation summary, set `ASURA_LANGSMITH_TRACE=1`, configure `LANGSMITH_API_KEY` in an environment/secret store, then call:

```python
from asura.integrations.langsmith import log_eval_metrics

log_eval_metrics({"val_loss": 1.2, "acc_all": 0.8}, data_policy="synthetic")
```

The required policy must be `synthetic` or `public`; only finite numeric metrics and that label are sent. Prompt text, generated samples, checkpoints, and dataset paths are never included. No credentials belong in source control or chat.

### Ollama Cloud comparison model

The optional Ollama client uses the OpenAI-compatible endpoint at `https://ollama.com/v1`,
with `deepseek-v4.1-flash` as its default model. Create an API key at
[ollama.com/settings/keys](https://ollama.com/settings/keys), then provide it through
`OLLAMA_API_KEY` (never commit it or paste it into chat):

```powershell
$env:OLLAMA_API_KEY = "<your Ollama API key>"
```

Only explicitly public or synthetic inputs are accepted because message content is sent to
Ollama's hosted API:

```python
from asura.integrations.ollama import chat_completion

reply = chat_completion(
    [{"role": "user", "content": "Does this informal example preserve its meaning?"}],
    data_policy="synthetic",
)
```

This is a hosted comparison/evaluation API, not the local vLLM inference engine. The client
is disabled unless called explicitly and does not log prompts or API keys.

### lm-evaluation-harness

Install the optional dependencies above, then load the plugin module explicitly:

```powershell
lm-eval run --model asura --model_args pretrained=runs/smoke/seed0/ckpt.pt,device=auto --tasks hellaswag --plugins asura.integrations.lm_eval
```

The adapter implements the harness model interface directly over Asura checkpoints and its byte tokenizer; it does not convert the model or require Transformers. Asura has no BOS/EOS token, so the adapter uses byte `0` (NUL) as a prefix sentinel for empty-context likelihoods. Context length is measured in UTF-8 bytes, not words or tokenizer pieces, so standard benchmarks can be slow and scores should be interpreted accordingly.

### ARC-AGI benchmarking

Asura can evaluate a local checkpoint on ARC-AGI task JSON files and write submission files compatible with the
[AXXESS TRIaxis ARC-AGI benchmarking fork](https://github.com/axxess-triaxis/arc-agi-benchmarking). This adapter
does not download benchmark datasets or require the fork's API-provider dependencies. Obtain task data separately
and use only splits you are permitted to evaluate:

```powershell
python -m asura.integrations.arc_agi `
  --checkpoint runs\smoke\seed0\ckpt.pt `
  --tasks-dir path\to\arc-agi-benchmarking\data\sample\tasks `
  --output-dir runs\arc-agi\asura-smoke
```

The adapter exposes training examples and test inputs only; expected test outputs are used locally for accuracy
when the task files include them. It writes one JSON submission per task in the fork's standard `attempt_1` schema,
which can then be scored with that repository's `src/arc_agi_benchmarking/scoring/scoring.py`. Asura uses a
byte-level tokenizer, so prompts longer than the checkpoint's `context_len` fail explicitly; train a checkpoint
with a larger context for full ARC tasks. Invalid model output is retained as text and counted in `invalid_outputs`
instead of being silently treated as a grid.
