# Paper → experiment worksheet

Copy this to `papers/<paper-short-name>/PLAN.md` for each of your 3 papers.

## 1. Paper
- Title:
- One-sentence thesis:

## 2. Claims (list every claim that could be tested)
| # | Claim (exact wording from paper) | Testable at small scale? | Why / why not |
|---|----------------------------------|--------------------------|---------------|
| C1 |  |  |  |

## 3. Pick ONE claim first. Turn it into a falsifiable hypothesis
- **Hypothesis:** If we change ___ (independent variable), then ___ (metric) will ___ (direction, rough size).
- **Null result would look like:**
- **Control condition:**
- **Everything held fixed:** data, steps, params (within ±10%), LR schedule, seeds.

## 4. Setup in Asura
- Data / task (existing task, or a new one in `asura/data.py`):
- Model change (new flag in `asura/model.py`, default OFF so baseline is unchanged):
- Eval (existing metric, or a new one in `asura/eval.py`):
- Sweep file: `experiments/configs/<name>.json`
- Seeds: ≥3 to start, ≥5 before you claim anything.

## 5. Results
| Condition | metric mean ± 95% CI | n seeds | params |
|-----------|----------------------|---------|--------|

## 6. Verdict
- Supported / not supported / inconclusive:
- Threats to validity (scale, task choice, tuning budget per condition, etc.):
- What would change my mind:
- Next experiment:
