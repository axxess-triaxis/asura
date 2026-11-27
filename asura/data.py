"""Datasets for Asura.

Two kinds:
  1. TextData   - any .txt file / folder (language modeling, perplexity evals)
  2. Synthetic tasks - fully controlled, so you can test a research claim
     with a known ground truth (addition, modular arithmetic, copy, reverse,
     sorting). Each example is "PROMPT=ANSWER\n"; loss is only on the answer.

To test a paper's claim, the cleanest move is usually to add a new task here
that isolates the mechanism the paper talks about.
"""
import random
from pathlib import Path
import torch
from .tokenizer import ByteTokenizer

TOK = ByteTokenizer()


class TextData:
    def __init__(self, path, val_frac=0.1):
        p = Path(path)
        files = sorted(p.rglob("*.txt")) if p.is_dir() else [p]
        text = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files)
        ids = torch.tensor(TOK.encode(text), dtype=torch.long)
        n = int(len(ids) * (1 - val_frac))
        self.train, self.val = ids[:n], ids[n:]

    def batch(self, split, batch_size, context_len, gen):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - context_len - 1, (batch_size,), generator=gen)
        x = torch.stack([data[i:i + context_len] for i in ix])
        y = torch.stack([data[i + 1:i + context_len + 1] for i in ix])
        return x, y, None


# ---------- synthetic tasks: each returns (prompt, answer) ----------

def task_add(rng, digits=3):
    a, b = rng.randrange(10 ** digits), rng.randrange(10 ** digits)
    return f"{a}+{b}=", str(a + b)

def task_mod(rng, p=97):
    a, b = rng.randrange(p), rng.randrange(p)
    return f"{a}*{b}%{p}=", str((a * b) % p)

def _letters(rng, lo=3, hi=10):
    return "".join(rng.choice("abcdefghij") for _ in range(rng.randint(lo, hi)))

def task_copy(rng):
    s = _letters(rng); return f"copy:{s}=", s

def task_reverse(rng):
    s = _letters(rng); return f"rev:{s}=", s[::-1]

def task_sort(rng):
    s = _letters(rng); return f"sort:{s}=", "".join(sorted(s))

TASKS = {"add": task_add, "mod": task_mod, "copy": task_copy,
         "reverse": task_reverse, "sort": task_sort}


# ---------- "messy world" rendering (Clean Data Paradox experiments) ----------
# Same underlying fact, rendered the way real records often look: Assamese/
# Bengali digits, stray spaces, words instead of symbols, leading zeros.
ASSAMESE_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

def messify(prompt, rng):
    ops = [lambda s: s.translate(ASSAMESE_DIGITS),
           lambda s: s.replace("+", " + ").replace("*", " * "),
           lambda s: s.replace("+", " plus "),
           lambda s: "".join(("0" + ch) if ch.isdigit() and i == 0 else ch for i, ch in enumerate(s))]
    for op in rng.sample(ops, rng.randint(1, 3)):
        prompt = op(prompt)
    return prompt


def corrupt(ans, rng):
    """Random label noise: a plausible-looking wrong number."""
    if not ans.isdigit():
        return ans[::-1] if len(ans) > 1 else ans
    wrong = int(ans) + rng.choice([-1, 1]) * rng.randint(1, 20)
    return str(abs(wrong))


class SyntheticData:
    """Fixed train/test split of generated examples (so memorization vs.
    generalization is measurable: test prompts are never seen in training).

    Messy-data knobs (train split only; test labels are always correct):
      messy_frac        fraction of train prompts rendered messily
      label_noise       fraction of train answers replaced with random wrong ones
      systematic_noise  fraction of *messy* train examples whose first operand is
                        even that get answer+1 -- a consistent, "motivated" error
                        tied to a subgroup, as opposed to random noise
    Test set: half clean, half messy, reported separately (acc_add vs acc_add~messy).
    """

    def __init__(self, tasks, n_train=20000, n_test=1000, seed=0,
                 messy_frac=0.0, label_noise=0.0, systematic_noise=0.0, **task_kwargs):
        rng = random.Random(seed)
        self.tasks = tasks
        seen, train, test = set(), [], []
        while len(train) + len(test) < n_train + n_test:
            name = rng.choice(tasks)
            fn = TASKS[name]
            kw = {k: v for k, v in task_kwargs.items() if k in fn.__code__.co_varnames}
            prompt, ans = fn(rng, **kw)
            if prompt in seen:          # dedupe on the canonical fact, not its rendering
                continue
            seen.add(prompt)
            first_even = prompt[0].isdigit() and int(prompt.split("+")[0].split("*")[0]) % 2 == 0
            is_test = len(test) < n_test and rng.random() < n_test / (n_train + n_test)
            if is_test:
                # messy test items (numeric tasks only), split by the subgroup the systematic error targets
                if any(ch.isdigit() for ch in prompt) and rng.random() < 0.5:
                    test.append((name + "~messy" + ("~even" if first_even else "~odd"), messify(prompt, rng), ans))
                else:
                    test.append((name, prompt, ans))
                continue
            messy = rng.random() < messy_frac
            p = messify(prompt, rng) if messy else prompt
            a = ans
            if messy and first_even and ans.isdigit() and rng.random() < systematic_noise:
                a = str(int(ans) + 1)
            elif rng.random() < label_noise:
                a = corrupt(ans, rng)
            train.append((name + ("~messy" if messy else ""), p, a))
        self.train, self.test = train, test

    def batch(self, split, batch_size, context_len, gen):
        pool = self.train if split == "train" else self.test
        ix = torch.randint(len(pool), (batch_size,), generator=gen).tolist()
        xs, ys, ms = [], [], []
        for i in ix:
            _, prompt, ans = pool[i]
            p, a = TOK.encode(prompt), TOK.encode(ans + "\n")
            seq = (p + a)[: context_len + 1]
            mask = ([0] * len(p) + [1] * len(a))[: context_len + 1]
            pad = context_len + 1 - len(seq)
            seq, mask = seq + [0] * pad, mask + [0] * pad
            xs.append(seq[:-1]); ys.append(seq[1:]); ms.append(mask[1:])
        t = lambda v: torch.tensor(v, dtype=torch.long)
        return t(xs), t(ys), t(ms)


def make_data(cfg: dict):
    if cfg["type"] == "text":
        return TextData(cfg["path"], cfg.get("val_frac", 0.1))
    if cfg["type"] == "synthetic":
        kw = {k: v for k, v in cfg.items() if k not in ("type", "tasks")}
        return SyntheticData(cfg["tasks"], **kw)
    raise ValueError(cfg["type"])
