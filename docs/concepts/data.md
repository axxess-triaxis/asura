# Tokenizer & data

## Byte-level tokenizer
`asura/tokenizer.py`: every UTF-8 byte is one token, so the vocabulary is 256.

```python
>>> from asura import ByteTokenizer
>>> t = ByteTokenizer()
>>> t.encode("12+3")
[49, 50, 43, 51]
>>> t.encode("১")          # Assamese/Bengali digit one: 3 bytes
[224, 167, 167]
```

**Pros:** no training, any script works, nothing is ever out-of-vocabulary.
**Cons:** non-Latin text costs 2-4× more tokens, so the context fills faster. A BPE/SentencePiece
tokenizer fixes that (roadmap Stage 3). Note the asymmetry, though: for a byte-level model,
Assamese is *literally* more expensive than English. That's a small, concrete instance of the
data-asymmetry argument in the Clean Data paper.

## Text datasets
```json
"data": {"type": "text", "path": "data/my_corpus", "val_frac": 0.1}
```
`path` is a `.txt` file or a folder (all `*.txt` files, recursively). The last 10% is held out for
validation. Batches are random windows of `context_len` bytes.

## Synthetic tasks
Synthetic tasks give **ground truth**, so a claim can be tested without arguing about labels.

| Task | Prompt → answer |
|---|---|
| `add` | `47+38=` → `85` (`digits` controls size) |
| `mod` | `12*30%97=` → `69` |
| `copy` | `copy:abcd=` → `abcd` |
| `reverse` | `rev:abcd=` → `dcba` |
| `sort` | `sort:dbca=` → `abcd` |

```json
"data": {"type": "synthetic", "tasks": ["add", "reverse"], "digits": 2,
         "n_train": 8000, "n_test": 400, "seed": 0}
```

- Loss is computed **only on the answer** (`loss_mask`), not the prompt.
- The test split holds out *facts*: a test prompt never appears in training in any rendering,
  so test accuracy measures generalization, not memorization.

## "Messy world" knobs
Added for the [Clean Data Paradox](../research/clean-data-paradox.md) experiments. They apply to the
training split only; test labels are always correct.

| Knob | Effect |
|---|---|
| `messy_frac` | Fraction of training prompts rendered messily: Assamese digits (`৪৭`), `plus` instead of `+`, stray spaces, leading zeros |
| `label_noise` | Fraction of answers replaced with a random plausible wrong number |
| `systematic_noise` | Fraction of *messy* examples with an even first operand whose answer becomes `answer+1`, a consistent, subgroup-targeted distortion |

The test set is half clean, half messy. Messy items are tagged `~even` / `~odd` so the effect of the
systematic distortion can be measured on exactly the subgroup it targets.

```text
('add~messy', '০৭৯ plus ৫৯=', '138')
('add~messy', '51 + 38=', '89')
('add',       '79+32=', '111')
```
