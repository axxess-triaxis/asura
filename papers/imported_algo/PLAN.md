# The Imported Algo and the Chowk (Chor) Bazaar: experiment plan + results

**Thesis:** Credit models trained on OECD data read informality (cash, family control,
political adjacency, thin documentation) as risk. Deployed in India, they will reject
legitimate informal businesses. Localized models that use behavioural weak signals can
tell informality apart from deception.

**Method:** a simulated two-region lending world (`experiments/imported_algo/sim.py`),
logistic-regression underwriter, approve the 70% least risky, 5 seeds, 95% CIs.
⚠️ The data generator *encodes* the paper's assumptions. This tests whether the argument
is internally coherent and which parts of it matter. It does not test whether India's
real credit data looks like this. That needs real data (see "Next").

## Claims tested

| # | Claim | Operationalized as | Result |
|---|-------|--------------------|--------|
| H1 | Imported models wrongly reject legitimate informal borrowers | Wrong-reject rate among non-defaulting informal borrowers, imported vs. localized | **Supported** (when behavioural features exist): 38.5% ± 1.1 vs 28.5% ± 1.3 |
| H1b | (implicit) the harm is invisible to efficiency metrics | Overall AUC and approved-pool default rate | **Supported**: AUC 0.764 vs 0.780, default rate 3.9% vs 3.7%. Nearly identical "efficiency", but the rejection burden shifts from formal (8.8%) to informal (38.5%) borrowers |
| H2 | Weak behavioural signals separate informality from deception | AUC *within* the informal subgroup, with vs. without behavioural features | **Supported, but partly by construction**: 0.51 → 0.80. The sim assumes deceptive borrowers have erratic behaviour. That assumption is the real empirical question |
| H2b | Localization alone fixes the problem | Localized model with formal features only | **Not supported**: localized ≈ imported (both 48.6% wrong-reject). Without behavioural data, localization buys nothing. *A finding worth adding to the paper* |
| H3 | Self-reinforcing loop: flagged groups stay flagged because the training data validates the suspicion | Retrain each round on approved borrowers' outcomes; with vs. without 10% exploration | **Not supported as stated.** Bias *decays slowly* (38% → 32% over 8 rounds) and exploration barely helps, because approved informal borrowers still generate corrective labels. What persists is the old OECD prior in the pooled data. Persistence came from data inertia, not a runaway loop |

## What this means for the paper
- The strongest version of the argument holds up: the harm is distributional and hidden
  behind aggregate accuracy. Say that explicitly. It's the most defensible and policy-relevant point.
- Sharpen the remedy. "Train on Indian data" is not enough. The fix needs *behavioural data
  plus local labels*. Localization without new signals did nothing here.
- Soften "self-reinforcing" to "persistent", or name the conditions under which it truly
  runs away: e.g. a group rejected at ~100% (selective labels), or flags that themselves
  *cause* default (rejected businesses go to costlier informal credit). These are next experiments.

## Next
1. **Runaway-loop conditions:** make rejection *causal* (a rejected business faces higher
   default risk next round) and drop OECD data after round 0. Does the bias then grow?
2. **Fairness constraints:** equalize wrong-reject rates across groups. What does it cost in default rate?
3. **Explainability claim:** compare per-borrower explanations (feature contributions) between the
   imported and localized models. How often would "cash" be the stated reason for rejecting a borrower who never defaults?
4. **Real data:** public proxies: RBI/CMIE aggregates, Kaggle Indian MSME loan sets, or a partnership
   with an NBFC/MFI for anonymized outcomes (the paper's own recommendation).
