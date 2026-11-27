# Why AI Trained on Clean Data Struggles (Clean Data Paradox): experiment plan

**Thesis:** Models trained only on clean, formal data fail on how the world actually
looks. Including messy, ambiguous data teaches the model "what should not be copied",
so it learns to extract the usable signal instead of reproducing the mess.

## Claims → hypotheses
| # | Claim | Hypothesis in Asura |
|---|-------|---------------------|
| C1 | Clean-only training fails on messy real-world inputs | A model trained on clean `a+b=` fails on the same facts rendered messily (Assamese digits, "plus", stray spaces, leading zeros) |
| C2 | Including messy data fixes this without hurting clean performance | 50% messy training data → high accuracy on *both* clean and messy held-out prompts |
| C3 | "Not garbage in, garbage out": the model learns to separate usable from unusable | With 20% *random* wrong labels, greedy outputs stay correct (the model learns the signal, not the noise) |
| C3b | (the paper's caution) politically motivated / spurious-on-purpose data gets absorbed as truth | With a *systematic* error (answer+1 on one subgroup), the model **reproduces** the bias on that subgroup, unlike random noise |

C3 vs C3b is the sharpest scientific point. Separating signal from noise only works when the
noise is unstructured. Consistent, motivated distortion looks like signal. If the results
show this, it's a precise, testable restatement of the paper's argument.

## Setup
- Task: 2-digit addition, held-out *facts* (test pairs never seen in any rendering).
- Model: Asura 0.36M params, 1500 steps. 3 seeds per condition.
- Config: `experiments/configs/clean_data_paradox.json`
- Metrics: `acc_add` (clean), `acc_add~messy~odd` / `~even` (messy, split by the targeted subgroup),
  `plus1_add~messy~even` (how often the model outputs the biased answer).

## Results
_Run in progress → `runs/clean_data_paradox.log`, `runs/clean_data_paradox/summary.json`._

## Next
- Dose-response: messy_frac ∈ {0, 0.1, 0.25, 0.5, 0.75}. Is there a sweet spot?
- Noise-rate curve: label_noise ∈ {0, 0.1, 0.2, 0.3, 0.45}. Where does "learn the signal" break?
- "Statistically fact-check itself" (Build Strategy #5): train on two "regions" with different
  error patterns. Does agreement across sources let the model recover the truth when each alone is biased?
- Scale up: real code-mixed text (Assamese/English) from public corpora (e.g. IndicCorp, AI4Bharat). Perplexity
  on informal vs. formal registers for a clean-trained vs. mixed-trained model.
