"""Simulation for "The Imported Algo and the Chowk (Chor) Bazaar".

What this tests: whether the *mechanisms* the paper describes hold up in a
controlled world we build ourselves. It does NOT show that real Indian lending
data behaves this way. The data generator encodes the paper's assumptions, and
the open question is what follows from them.

World: two regions with different causal structure.
  OECD : informality (cash, low docs, family control) is rare and is mostly
         a symptom of deception, so it predicts default.
  India: informality is common and mostly benign. Default is driven by
         deception and by weak behavioural signals (repayment rhythm, utility
         consistency, procurement stability). Deceptive borrowers look informal
         but their behavioural signals are erratic.

Hypotheses (from the paper):
  H1 An imported model (trained on OECD data) wrongly rejects legitimate
     informal Indian borrowers far more often than a localized model does.
  H2 Behavioural "weak signals" let a model tell informality apart from
     deception: AUC inside the informal subgroup jumps when they are included.
  H3 Feedback loop: if only approved borrowers generate repayment labels,
     retraining on deployment data does not fix the imported bias (or makes
     it worse). A little exploration (approving a random slice of rejects)
     does fix it.

Run:  .venv\\Scripts\\python experiments\\imported_algo\\sim.py --seeds 5
"""
import argparse, json, statistics
from pathlib import Path
import torch

FORMAL = ["cash", "low_docs", "family", "political"]
BEHAV = ["rhythm", "utility", "procure"]
ALL = FORMAL + BEHAV


def gen(region, n, g):
    r = lambda *s: torch.rand(*s, generator=g)
    deceptive = (r(n) < 0.06).float()
    if region == "oecd":
        informal = torch.where(deceptive.bool(), (r(n) < 0.8).float(), (r(n) < 0.08).float())
    else:  # india
        informal = torch.where(deceptive.bool(), (r(n) < 0.85).float(), (r(n) < 0.60).float())
    cash = (0.25 + 0.5 * informal + 0.15 * torch.randn(n, generator=g)).clamp(0, 1)
    low_docs = (0.2 + 0.55 * informal + 0.15 * torch.randn(n, generator=g)).clamp(0, 1)
    family = (r(n) < 0.3 + 0.5 * informal).float()
    political = (r(n) < 0.05 + 0.15 * informal).float()
    # behavioural signals: high & stable for honest businesses (formal or not), erratic for deceptive
    quality = torch.randn(n, generator=g) * 0.6          # honest business quality varies
    base = 0.7 + 0.15 * quality - 0.45 * deceptive
    noise = lambda: 0.12 * torch.randn(n, generator=g)
    rhythm, utility, procure = ((base + noise()).clamp(0, 1) for _ in range(3))
    logit = -3.2 + 3.0 * deceptive - 1.6 * quality
    if region == "oecd":                                  # in OECD, informality itself carries risk
        logit = logit + 1.0 * informal
    default = (r(n) < torch.sigmoid(logit)).float()
    X = torch.stack([cash, low_docs, family, political, rhythm, utility, procure], 1)
    return {"X": X, "y": default, "informal": informal, "deceptive": deceptive}


def fit(X, y, l2=1e-3, steps=300):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b, y) + l2 * (w ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    w, b = w.detach(), b.detach()
    return lambda Z: torch.sigmoid(((Z - mu) / sd) @ w + b)


def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return (pos[:, None] > neg[None, :]).float().mean().item()


def cols(d, names):
    return d["X"][:, [ALL.index(c) for c in names]]


def report(model, d, names, approve_rate=0.7):
    s = model(cols(d, names))
    thr = s.quantile(approve_rate)             # approve the 70% least risky
    reject = s > thr
    good = d["y"] == 0
    legit_inf = good & (d["informal"] == 1)
    legit_for = good & (d["informal"] == 0)
    inf = d["informal"] == 1
    return {
        "auc": auc(s, d["y"]),
        "auc_within_informal": auc(s[inf], d["y"][inf]),
        "wrong_reject_legit_informal": reject[legit_inf].float().mean().item(),
        "wrong_reject_legit_formal": reject[legit_for].float().mean().item(),
        "default_rate_of_approved": d["y"][~reject].mean().item(),
    }


def feedback_loop(seed, rounds=8, per_round=4000, explore=0.0, names=ALL):
    """Deploy the imported model in India. Each round, only approved borrowers
    (plus a random `explore` slice of rejects) reveal whether they defaulted.
    Retrain on OECD + all revealed Indian labels."""
    g = torch.Generator().manual_seed(1000 + seed)
    oecd = gen("oecd", 20000, g)
    test = gen("india", 20000, g)
    Xs, ys = [cols(oecd, names)], [oecd["y"]]
    model = fit(Xs[0], ys[0])
    hist = []
    for t in range(rounds):
        hist.append({"round": t, **report(model, test, names)})
        d = gen("india", per_round, g)
        s = model(cols(d, names))
        approved = s <= s.quantile(0.7)
        lucky = (torch.rand(per_round, generator=g) < explore) & ~approved
        seen = approved | lucky
        Xs.append(cols(d, names)[seen]); ys.append(d["y"][seen])
        model = fit(torch.cat(Xs), torch.cat(ys))
    return hist


def one_seed(seed):
    g = torch.Generator().manual_seed(seed)
    oecd, india_tr, india_te = gen("oecd", 20000, g), gen("india", 20000, g), gen("india", 20000, g)
    out = {}
    for tag, names in [("formal_only", FORMAL), ("formal+behav", ALL)]:
        out[f"imported/{tag}"] = report(fit(cols(oecd, names), oecd["y"]), india_te, names)
        out[f"localized/{tag}"] = report(fit(cols(india_tr, names), india_tr["y"]), india_te, names)
    out["oracle_frac_informal_india"] = india_te["informal"].mean().item()
    out["loop/no_explore"] = feedback_loop(seed, explore=0.0)
    out["loop/explore10"] = feedback_loop(seed, explore=0.10)
    return out


def ms(xs):
    m = statistics.mean(xs); s = statistics.stdev(xs) if len(xs) > 1 else 0
    return f"{m:.3f} ± {1.96 * s / len(xs) ** 0.5:.3f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    runs = [one_seed(s) for s in range(a.seeds)]
    outdir = Path("runs/imported_algo"); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(runs, indent=1))

    print(f"\n== H1/H2: static comparison on Indian test data (mean ± 95% CI, {a.seeds} seeds) ==")
    keys = ["auc", "auc_within_informal", "wrong_reject_legit_informal", "wrong_reject_legit_formal", "default_rate_of_approved"]
    print(f"{'model':28s}" + "".join(f"{k:>30s}" for k in keys))
    for m in ["imported/formal_only", "localized/formal_only", "imported/formal+behav", "localized/formal+behav"]:
        print(f"{m:28s}" + "".join(f"{ms([r[m][k] for r in runs]):>30s}" for k in keys))

    print("\n== H3: feedback loop, wrong-reject rate for legit informal borrowers by round ==")
    for loop in ["loop/no_explore", "loop/explore10"]:
        series = [ms([r[loop][t]["wrong_reject_legit_informal"] for r in runs]) for t in range(len(runs[0][loop]))]
        print(f"{loop:18s} " + "  ".join(series))
    print(f"\nraw results -> {outdir / 'results.json'}")


if __name__ == "__main__":
    main()
