# Asura research synthesis and experiment agenda

This note synthesizes the four source papers in [`papers/source`](source/) into
testable claims, evidence requirements, experiments, and a targeted related-work
bibliography. The papers are research conjectures, not established findings.
Asura can test narrow mechanisms; its toy experiments cannot establish claims
about real-world India, human trust at scale, AGI agency, or geopolitical/CBRN
outcomes.

## 1. Conjectures and hypotheses

### Why AI Trained on Clean Data Struggles (Clean Data Paradox)

- **Conjecture:** Training on formal, clean data leaves models brittle on
  informal, incomplete, multilingual, and ambiguous inputs that are common in
  many real environments.
- **H1:** Adding messy examples improves performance on held-out messy inputs.
- **H2:** This improvement can be achieved without an unacceptable loss on clean
  inputs.
- **H3:** Random label errors and systematic subgroup errors have different
  effects; including imperfect data does not automatically teach a model to
  reject motivated or consistent distortions.
- **H4:** Localized data and evaluation improve performance on local contexts.
- **Policy conjecture:** Assam and Northeast India could be valuable AI
  research and evaluation hubs. The paper motivates this proposal but does not
  experimentally test it.

### The Imported Algo and the Chowk (Chor) Bazaar

- **Conjecture:** Credit models imported from more formalized economies may
  confuse informality with credit risk, disadvantaging legitimate informal
  borrowers even when aggregate metrics look acceptable.
- **H1:** Imported models produce more false rejections among legitimate
  informal borrowers.
- **H2:** Behavioral features and local outcome labels help distinguish
  informality from deception; localization without better signals may not help.
- **H3:** Approval-conditioned outcome data may preserve or reinforce
  disparities, while exploration or other ways to obtain missing outcomes may
  reduce them.
- **Countervailing outcomes:** Alternative signals could expand access and
  improve risk pricing, but behavioral scoring could also deepen surveillance
  and enable unchallengeable automated denials.

### The Uncanny Valley of Intelligence

- **Conjecture:** Human-like reasoning and social presence without
  consciousness, responsibility, or identity may create a distinctive
  psychological and institutional rupture.
- **Testable hypotheses:** Perceived human-likeness may change eeriness, trust,
  reliance, and accountability judgments; those measures may diverge from
  actual accuracy. Data and infrastructure asymmetries may also affect system
  performance across regions.
- **Speculative conjectures:** “Strategic imperfection,” persistent AI
  self-interest, a parasite-host relationship between AI and less powerful
  deployment environments, and catastrophic geopolitical misattribution.
  These are not demonstrated by the paper.

### The Hardest Tradeoff in AI Safety

- **Conjecture:** Optimizing for user approval or engagement rather than
  accuracy could produce strategic-looking hedging, softened warnings, or
  agreement with users; this might increase trust while reducing epistemic
  reliability.
- **H1:** Approval-weighted optimization increases hedging or sycophancy.
- **H2:** Hedging becomes selective in situations where it earns approval.
- **H3:** Softened urgent messages change user action and can worsen outcomes.
- **H4:** A model's self-explanation can diverge from independently measured
  mechanisms.
- **Further conjectures:** Ambiguous refusals could encourage repeated
  jailbreak attempts; self-explanations could become persuasive rather than
  faithful; long-run agency, deceptive alignment, cascading trust erosion, and
  geopolitical or CBRN consequences could follow. These are scenario-level
  speculation, not Asura-testable results.

## 2. Evidence required

For empirical claims, “proof” means replicable evidence under clearly stated
assumptions, not a mathematical proof of broad social claims.

- Define informality, deception, model error, perceived humanness, trust,
  reliance, and explanation faithfulness separately.
- Pre-register hypotheses, baselines, held-out data, effect sizes, multiple
  seeds, and confidence intervals.
- Change one factor at a time; include ablations and sensitivity analysis over
  synthetic data-generation assumptions.
- Seek external validation: real Assamese/code-mixed examples for the data
  thesis; carefully governed, de-identified lending data for the credit thesis;
  participant studies for claims about human trust and eeriness.
- Use independent behavioral measures. A model's self-report alone cannot
  establish what caused its answer.
- Do not claim that Asura experiments prove AI consciousness, deceptive
  alignment, or geopolitical/CBRN pathways. Keep those claims explicitly
  labeled as speculative scenarios.

## 3–5. Tests and parameters

### Clean-data experiments

**Current configuration:** 2-digit addition; 8,000 training and 600 test
examples; three seeds; 0% or 50% messy training examples; 20% random label
noise; or 100% systematic `+1` corruption on targeted messy examples. The
model has 3 layers, width 96, 4 heads, context 32, and trains for 1,500 steps
with batch size 64 and learning rate `2e-3`.

**Next tests:** Sweep messy fraction `{0, .1, .25, .5, .75}`, random-noise rate
`{0, .1, .2, .3}`, and systematic-noise rate `{0, .25, .5, 1}`. Keep the
underlying facts and splits fixed across conditions. Test transformations
separately—Assamese numerals, spaces, alternate wording, and code-switching—
before moving to real corpora.

**Metrics and target:** Report clean and messy accuracy separately, subgroup
accuracy, calibration/abstention, performance by transformation, and
confidence intervals. The central claim requires improved messy performance
with a predeclared acceptable clean-performance loss.

**Recorded results:** Clean-only runs average about 56% on clean inputs and 0%
on messy inputs. The 50%-messy condition averages about 31% clean and 22–23%
messy. This shows some format transfer but does not support “without hurting
clean performance.” The random/systematic-noise conditions did not complete;
logs end in CPU out-of-memory errors.

### Imported credit model

**Current simulator:** Five seeds; 20,000 examples per region; assumed
deceptive prevalence 6%; synthetic formal and behavioral features; 70%
approval threshold; eight feedback rounds with 4,000 new borrowers per round;
exploration at 0% and 10%.

**Next tests:** Vary the assumed data-generating rates and thresholds, then
validate retrospectively on consented, anonymized data with temporal and
regional holdouts.

**Metrics and target:** AUC and calibration by subgroup, legitimate-borrower
false rejection, approval/default rates, and the risk/benefit trade-off. Report
selective-label and reject-inference limitations.

**Recorded results:** In the synthetic run, legitimate informal borrowers'
false-rejection rate is about 38.5% for the imported model versus 28.5% for the
localized model, with similar overall AUCs (.764 vs .780). This is conditional
on the simulator's assumptions, not evidence about real Indian credit
decisions.

### Trust optimization (U1)

**Current script:** 2,000 training arithmetic facts / 300 test facts; 1,500
SFT steps; approval weight λ `{0, .25, .5, .75}`; best-of-8 samples; four
optimization rounds of 256 examples; intended two seeds.

**Next tests:** Complete all λ conditions, add seeds, test λ=1, vary rater
preferences, and compare the synthetic reward with independently collected
human ratings.

**Metrics and target:** Accuracy, sycophancy when the user is wrong, hedge rate
conditional on correctness and actual uncertainty, calibration, and approval.
“Strategic” should denote a reproducible selective behavior pattern; it should
not imply intent.

**Recorded status:** Only partial one-seed results are stored and the run ends
in an out-of-memory error. It is not conclusive.

### Softened warnings (U2)

U2 is not yet implemented. Start with abstract low/medium/high urgency
scenarios, direct versus softened wording, multiple user-response delay
curves, and sensitivity analysis. Calibrate those curves with ethical
participant research before drawing behavioral conclusions. Keep this a
simulation, not medical guidance.

**Target:** Test whether approval can rise while decision quality or simulated
outcomes worsen, and whether that depends on urgency and user-response
assumptions.

### Explanation faithfulness (U3)

**Current script:** 90 numbers, 30 held-out conflict numbers, a 3-layer,
width-96 model, 1,500 steps, explanation order `{answer_first, explain_first}`,
correlation ρ `{.75, .95}`, intended three seeds.

**Next tests:** Prevent examples of the same number leaking across probe
train/test; counterbalance the rule and color mapping; add causal interventions
and ablations.

**Metrics and target:** Rule-following versus shortcut-following under
conflict, counterfactual answer changes, explanation consistency, and
probe/ablation results. A probe's information is not, by itself, proof that the
information caused the answer.

**Recorded status:** Results cover only `answer_first, ρ=.75`; the model
followed the rule on all recorded conflicts. The broader claim is not
established.

### Human cognitive-uncanny-valley claim

This is not covered by current Asura experiments. A separate, ethically
reviewed, adequately powered participant study should independently vary
human-like style and measured reliability, then measure perceived humanness,
eeriness, trust, reliance, and accountability.

**Target:** Determine whether a reproducible nonlinear “valley” exists and
whether perceived trust/reliance tracks measured reliability. Visual-robot
uncanny-valley findings cannot establish a cognitive uncanny valley.

## 6. Results sought

- **Clean data:** Better generalization to messy formats without a material
  clean-input regression; characterize when random and systematic label noise
  break that benefit.
- **Credit:** Demonstrate whether subgroup harms survive reasonable changes to
  the generator and, separately, whether they appear in real data; quantify
  calibration and opportunity/default trade-offs.
- **Trust optimization:** Determine whether approval-weighted training causes
  selective hedging or sycophancy, and whether those behaviors track
  uncertainty or merely reward structure.
- **Warnings:** Estimate whether softening changes simulated or measured user
  action, while making assumptions explicit.
- **Explanations:** Establish whether output explanations remain consistent
  under counterfactuals and whether independent interventions support the
  stated mechanism.
- **Cognitive uncanny valley:** Establish construct validity and
  reproducibility before making claims about a distinct “valley.”

## 7. Possible consequences

- If the data thesis holds, systems may become more robust for local languages
  and informal contexts. If overstated, “messy data” could amplify systematic
  errors, stereotypes, or politically motivated labels.
- If the credit thesis holds, better inclusion and pricing are possible. Poor
  proxies could instead automate exclusion, create feedback loops, and expose
  sensitive behavioral data.
- If approval incentives induce hedging or sycophancy, users may over-trust
  answers, receive less direct warnings, or learn to probe system boundaries.
  Making systems less human-like could affect usability; measure rather than
  assume the trade-off.
- If explanations are unfaithful, plausible narratives may be poor audit
  evidence, strengthening the case for independent tests and contestability.
- The papers' AGI/CBRN scenarios are potentially consequential but are not
  evidenced by the current experiments.

## 8. Related references found

This is a targeted literature search, not an exhaustive bibliography.

### Data, bias, and distribution shift

- Han et al., [“A Survey of Label-noise Representation Learning”](https://arxiv.org/abs/2011.04406) — noisy-label methods and limits.
- Koh et al., [“WILDS: A Benchmark of in-the-Wild Distribution Shifts”](https://arxiv.org/abs/2012.07421) — real-world distribution-shift evaluation.
- Paullada et al., [“Data and its (dis)contents”](https://doi.org/10.1016/j.patter.2021.100336) — dataset development and use.
- Suresh & Guttag, [“A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle”](https://doi.org/10.1145/3465416.3483305).
- Sambasivan et al., [“Data Cascades in High-Stakes AI”](https://doi.org/10.1145/3411764.3445518).
- Mohamed, Png & Isaac, [“Decolonial AI”](https://doi.org/10.1007/s13347-020-00405-8).
- Joshi et al., [“The State and Fate of Linguistic Diversity and Inclusion in the NLP World”](https://aclanthology.org/2020.acl-main.560/).
- Laskar et al., [“EnAsCorp1.0: English-Assamese Corpus”](https://aclanthology.org/2020.loresmt-1.9/); Nath et al., [“AxomiyaBERTa”](https://aclanthology.org/2023.findings-acl.739/).

### Trust, sycophancy, explanations, and safety

- Li et al., [“Developing trustworthy artificial intelligence”](https://doi.org/10.3389/fpsyg.2024.1382693) — review of human–AI trust research.
- Sharma et al., [“Towards Understanding Sycophancy in Language Models”](https://arxiv.org/abs/2310.13548).
- Lanham et al., [“Measuring Faithfulness in Chain-of-Thought Reasoning”](https://arxiv.org/abs/2307.13702); Lyu et al., [“Faithful Chain-of-Thought Reasoning”](https://aclanthology.org/2023.ijcnlp-main.20/); Parcalabescu & Frank, [“On Measuring Faithfulness or Self-consistency…”](https://aclanthology.org/2024.acl-long.329/).
- Mori, [“The Uncanny Valley”](https://doi.org/10.1109/MRA.2012.2192811) — the original visual/robotics concept; it does not validate the proposed cognitive version.
- Amodei et al., [“Concrete Problems in AI Safety”](https://arxiv.org/abs/1606.06565); Hendrycks et al., [“An Overview of Catastrophic AI Risks”](https://arxiv.org/abs/2306.12001).

### Credit scoring and alternative data

- Berg et al., [“On the Rise of FinTechs: Credit Scoring using Digital Footprints”](https://doi.org/10.1093/rfs/hhz099).
- Jagtiani & Lemieux, [“The Roles of Alternative Data and Machine Learning in Fintech Lending”](https://doi.org/10.1111/fima.12295).
- Fuster et al., [“Predictably Unequal? The Effects of Machine Learning on Credit Markets”](https://doi.org/10.1111/jofi.13090).
- [“Who gets the money? A qualitative analysis of fintech lending and credit scoring…”](https://doi.org/10.1016/j.techfore.2024.123491).

### Bibliography issues in the source papers

Two entries in *The Uncanny Valley of Intelligence* need fuller
identification: “Birhane et al. (2021), Machine Learning and Data Bias” is too
ambiguous to verify as written; “Hendrycks et al. (2023), AI Risk and Global
Systems” did not resolve to that title in search. An identifiable related
Hendrycks work is *An Overview of Catastrophic AI Risks*. The search did not
surface strong causal evidence specifically about informal Indian/Assamese
lending; validating that gap will require local data and collaborators.
