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
