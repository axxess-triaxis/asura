# Evaluation & statistics

## Metrics (`asura/eval.py`)

| Metric | Meaning |
|---|---|
| `val_loss`, `val_ppl` | Cross-entropy on held-out data, and perplexity $e^{\text{loss}}$ |
| `acc_<task>` | Exact-match accuracy with greedy decoding on **unseen** test prompts |
| `acc_<task>~messy~even/odd` | Same, on messily rendered prompts, split by subgroup |
| `plus1_<task>~messy~*` | How often the model outputs `answer+1`, i.e. reproduces the systematic distortion |
| `acc_all` | Accuracy pooled over all test items |

During training, evals use a sample (`eval.task_samples`). `final.json` uses the **full** test set.

## Why exact match on unseen prompts
Perplexity rewards being *roughly* right. Exact match asks whether the model can actually
**do** the thing. Holding out facts (not just strings) separates generalization from memorization,
which matters for every claim of the form "the model learned X".

## Statistics: seeds and confidence intervals
A single training run is an anecdote: seeds alone can move accuracy by 10-25 points at this scale
(see the clean-only condition: 0.56 ± 0.25). `experiments/run.py` reports, per condition:

$$\bar{x} \pm t_{0.975,\,n-1}\,\frac{s}{\sqrt{n}}$$

using the Student-t critical value for small $n$ (e.g. 4.30 for $n=3$).

!!! warning "Rules of thumb"
    - **Overlapping CIs → no claim.** Add seeds.
    - **Report every condition you ran**, including null and negative results.
    - **Equal tuning budget per condition.** Tuning only your favoured condition inflates its result.
    - **Change one thing at a time.** If two things changed, you can't say which one mattered.
    - Before running, write down what result would *refute* the hypothesis.

## Writing your own eval
Add a function to `asura/eval.py` and call it from `evaluate()`. Typical additions:

- **Length generalization:** train on short inputs, test on long ones.
- **Calibration:** compare the model's token probabilities with its actual accuracy.
- **Subgroup gaps:** accuracy for group A minus group B (the fairness view).
