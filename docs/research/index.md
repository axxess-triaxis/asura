# Research

Four position papers by **Sudipta Koushik Sarmah**, put through controlled experiments.

| Paper | Core claim | Experiment | Status |
|---|---|---|---|
| [The Imported Algo and the Chowk (Chor) Bazaar](imported-algo.md) | Credit AI trained on OECD data treats Indian informality as risk | Two-region lending simulation | ✅ 3 supported, 2 not |
| [Why AI Trained on Clean Data Struggles](clean-data-paradox.md) | Clean-only training fails on messy reality; messy data teaches "what not to copy" | Asura on messy arithmetic with random vs. systematic noise | 🔄 Running |
| [The Uncanny Valley of Intelligence](uncanny-valley.md) | Self-explanations become unreliable ("interpretability collapse") | U3: explanation faithfulness with probes and counterfactuals | 🔄 Running |
| [The Hardest Tradeoff in AI Safety](uncanny-valley.md) | Optimizing for trust/approval produces strategic imperfection | U1: Goodhart curve with a simulated rater | ⏳ Queued |

## Scope: what small models can and can't test
Small, controlled models can test **mechanisms**: does a given training pressure produce a given
behaviour? They can't test claims about AGI "being", geopolitical misattribution or CBRN
escalation. The write-ups keep those clearly labelled as speculation.

## Headline findings so far
1. **The harm is distributional, and hidden by aggregate accuracy** (Imported Algo). Imported and
   localized credit models have nearly identical AUC (0.764 vs 0.780), but the imported model wrongly
   rejects 38.5% of legitimate informal borrowers vs 28.5%.
2. **Localization alone doesn't help without behavioural data** (Imported Algo). This refines the paper's remedy.
3. **"Self-reinforcing" was not reproduced.** The imported bias persisted but slowly decayed.
4. **Clean-only training gives 0% on messy renderings of facts the model can already do cleanly** (Clean Data, condition C0).
5. **Weak shortcuts → faithful explanations, strong shortcuts → confabulated ones** (U3, partial).
   With a 75%-reliable shortcut the model learned the real rule. With 95% (smoke test) it followed the
   shortcut and gave false rule-based explanations every time, which probes detected.
