"""Train Asura from a JSON config.

    python -m asura.train experiments/configs/smoke.json
    python -m asura.train experiments/configs/smoke.json --seed 1 --set model.d_model=128

Writes to runs/<name>/seed<k>/: config.json, log.jsonl, ckpt.pt, final.json
"""
import argparse, json, math, time, copy
from pathlib import Path
import torch
from .model import Asura, AsuraConfig
from .data import make_data
from .eval import evaluate
from .device import get_device, describe


def to_dev(batch, dev):
    return tuple(t.to(dev) if t is not None else None for t in batch)


def lr_at(step, cfg):
    warm, total, peak = cfg["warmup"], cfg["steps"], cfg["lr"]
    if step < warm:
        return peak * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return cfg.get("min_lr_frac", 0.1) * peak + (1 - cfg.get("min_lr_frac", 0.1)) * peak * 0.5 * (1 + math.cos(math.pi * prog))


def apply_overrides(cfg, sets):
    for s in sets or []:
        key, val = s.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for k in path:
            node = node.setdefault(k, {})
        try:
            node[last] = json.loads(val)
        except json.JSONDecodeError:
            node[last] = val
    return cfg


def train(cfg: dict, seed: int = 0, out_root="runs", verbose=True):
    cfg = copy.deepcopy(cfg)
    from .integrations.tracking import (
        log_mlflow_run,
        mlflow_enabled,
        require_mlflow,
        validate_local_tracking_uri,
    )
    tracking_enabled = mlflow_enabled(cfg)
    if tracking_enabled:
        validate_local_tracking_uri(cfg["tracking"].get("uri", "file:./mlruns"))
        require_mlflow()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tc = cfg["train"]
    out = Path(out_root) / cfg["name"] / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg, "seed": seed}, indent=2))

    data = make_data(cfg["data"])
    mcfg = AsuraConfig(**cfg.get("model", {}))
    dev = get_device()
    model = Asura(mcfg).to(dev)
    torch.set_num_threads(tc.get("threads", torch.get_num_threads()))

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tc.get("weight_decay", 0.1)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tc["lr"], betas=tuple(tc.get("betas", (0.9, 0.95))),
    )
    if verbose:
        print(f"[asura] {cfg['name']} seed={seed} params={model.num_params()/1e6:.2f}M device={describe(dev)} -> {out}")

    log = open(out / "log.jsonl", "w")
    t0 = time.time()
    for step in range(tc["steps"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        model.train()
        x, y, m = to_dev(data.batch("train", tc["batch_size"], mcfg.context_len, gen), dev)
        _, loss = model(x, y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
        opt.step()

        rec = None
        if step % tc.get("log_every", 50) == 0 or step == tc["steps"] - 1:
            rec = {"step": step, "loss": loss.item(), "lr": lr_at(step, tc),
                   "grad_norm": gnorm.item(), "elapsed_s": round(time.time() - t0, 1)}
        if step % tc.get("eval_every", 500) == 0 or step == tc["steps"] - 1:
            rec = rec or {"step": step}
            rec.update(evaluate(model, data, cfg.get("eval", {}), gen))
        if rec:
            log.write(json.dumps(rec) + "\n"); log.flush()
            if verbose:
                print("  " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    log.close()

    final = evaluate(model, data, {**cfg.get("eval", {}), "full": True}, gen)
    final.update({"params": model.num_params(), "train_time_s": round(time.time() - t0, 1), "seed": seed})
    (out / "final.json").write_text(json.dumps(final, indent=2))
    torch.save({"model": model.state_dict(), "model_cfg": mcfg.to_dict(), "cfg": cfg}, out / "ckpt.pt")
    if tracking_enabled:
        log_mlflow_run(cfg, final, seed)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", help="override, e.g. model.n_layers=4 train.lr=3e-4")
    a = ap.parse_args()
    cfg = apply_overrides(json.loads(Path(a.config).read_text()), a.set)
    print(json.dumps(train(cfg, a.seed), indent=2))


if __name__ == "__main__":
    main()
