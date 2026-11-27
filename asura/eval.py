"""Evaluations for Asura.

  - loss / perplexity on held-out data (works for any dataset)
  - exact-match accuracy per synthetic task, via greedy generation on
    *unseen* test prompts (measures generalization, not memorization)

Add your own eval here when a paper makes a claim about a specific behavior
(e.g. length generalization: train on short inputs, eval on long ones).
"""
import math
from collections import defaultdict
import torch
from .data import SyntheticData, TOK


@torch.no_grad()
def eval_loss(model, data, n_batches, batch_size, gen):
    model.eval()
    losses = []
    for _ in range(n_batches):
        dev = model.embed.weight.device
        x, y, m = (t.to(dev) if t is not None else None for t in data.batch("val", batch_size, model.cfg.context_len, gen))
        losses.append(model(x, y, m)[1].item())
    l = sum(losses) / len(losses)
    return {"val_loss": l, "val_ppl": math.exp(min(l, 20))}


@torch.no_grad()
def eval_tasks(model, data: SyntheticData, n, gen):
    model.eval()
    pool = data.test if n is None else data.test[:n]
    hits, tot, plus1 = defaultdict(int), defaultdict(int), defaultdict(int)
    for name, prompt, ans in pool:
        idx = torch.tensor([TOK.encode(prompt)], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=len(ans) + 4, temperature=0, stop_token=TOK.newline)
        pred = TOK.decode(out[0, idx.shape[1]:].tolist()).split("\n")[0]
        hits[name] += int(pred == ans); tot[name] += 1
        # did the model reproduce the systematic "+1" error? (Clean Data Paradox)
        plus1[name] += int(ans.isdigit() and pred == str(int(ans) + 1))
    res = {f"acc_{k}": hits[k] / tot[k] for k in tot}
    res.update({f"plus1_{k}": plus1[k] / tot[k] for k in tot if "~messy" in k})
    res["acc_all"] = sum(hits.values()) / max(1, sum(tot.values()))
    return res


def evaluate(model, data, ecfg, gen):
    full = ecfg.get("full", False)
    res = eval_loss(model, data, ecfg.get("loss_batches", 10) * (3 if full else 1), ecfg.get("batch_size", 64), gen)
    if isinstance(data, SyntheticData):
        res.update(eval_tasks(model, data, None if full else ecfg.get("task_samples", 200), gen))
    return res
