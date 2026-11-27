"""Run a controlled experiment: a grid of conditions x several seeds.

    python experiments/run.py experiments/configs/smoke_sweep.json

Sweep file format:
{
  "base": "experiments/configs/smoke.json",   # base training config
  "name": "depth_vs_width",
  "seeds": [0, 1, 2],
  "grid": {                                    # every combination is a condition
    "model.n_layers": [2, 4],
    "model.d_model": [64, 128]
  },
  "metric": "acc_all"                          # primary metric for the summary
}

Output: runs/<name>/summary.json and a printed table with mean +- std and a
95% CI per condition. Rule of thumb: if CIs overlap heavily, you have NOT shown
a difference; add seeds before claiming anything.
"""
import itertools, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asura.train import train, apply_overrides  # noqa: E402


def ci95(xs):
    if len(xs) < 2:
        return float("nan")
    # t critical values for small n (df = n-1)
    t = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}.get(len(xs) - 1, 1.96)
    return t * statistics.stdev(xs) / len(xs) ** 0.5


def main(path):
    sweep = json.loads(Path(path).read_text())
    base = json.loads(Path(sweep["base"]).read_text())
    # Either a full grid ("grid") or a list of hand-named conditions ("conditions": {label: {key: val}})
    if "conditions" in sweep:
        conds = [(label, [f"{k}={json.dumps(v)}" for k, v in ov.items()]) for label, ov in sweep["conditions"].items()]
    else:
        keys = list(sweep.get("grid", {}))
        combos = list(itertools.product(*sweep["grid"].values())) if keys else [()]
        conds = [(",".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, c)) or "base",
                  [f"{k}={json.dumps(v)}" for k, v in zip(keys, c)]) for c in combos]
    metric = sweep.get("metric", "val_loss")
    rows = []
    for label, sets in conds:
        cfg = apply_overrides(json.loads(json.dumps(base)), sets)
        cfg["name"] = f"{sweep['name']}/{label}"
        results = []
        for s in sweep["seeds"]:   # resume: reuse seeds that already finished
            done = Path("runs") / cfg["name"] / f"seed{s}" / "final.json"
            results.append(json.loads(done.read_text()) if done.exists()
                           else train(cfg, seed=s, verbose=sweep.get("verbose", False)))
        vals = [r[metric] for r in results]
        allm = {k: {"mean": statistics.mean([r[k] for r in results]), "ci95": ci95([r[k] for r in results])}
                for k in results[0] if isinstance(results[0][k], float)}
        row = {"condition": label, "n": len(vals), "mean": statistics.mean(vals),
               "std": statistics.stdev(vals) if len(vals) > 1 else 0.0, "ci95": ci95(vals),
               "values": vals, "params": results[0]["params"], "all_metrics": allm}
        rows.append(row)
        print(f"{label:40s} {metric}={row['mean']:.4f} +- {row['ci95']:.4f}  (n={row['n']}, params={row['params']/1e6:.2f}M)")
        for k in sweep.get("report", []):
            if k in allm:
                print(f"    {k:28s} {allm[k]['mean']:.4f} +- {allm[k]['ci95']:.4f}")
        sys.stdout.flush()
    out = Path("runs") / sweep["name"] / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sweep": sweep, "metric": metric, "rows": rows}, indent=2))
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
