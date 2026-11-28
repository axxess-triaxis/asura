# Assamese corpus pilot runs

**Run date:** 2026-11-27  
**Purpose:** Verify Asura's text-training path and establish an initial formal/web Assamese baseline. These runs do not evaluate informal Assamese.

## Data

The source was the `text` field of the [`MWirelabs/assamese-monolingual-corpus`](https://huggingface.co/datasets/MWirelabs/assamese-monolingual-corpus) training split. Its dataset card reports 1,613,879 Assamese sentences, declares **CC BY-SA 4.0**, and lists IITB-IndicMonoDoc, the Assamese side of Samanantar, Assamese poetry, and civic texts as sources. The observed samples were largely formal or web/news style.

The 100,000-sentence pilot was streamed with `shuffle(buffer_size=100_000)` and stopped after 100,000 rows. This locally shuffles the stream; it is not a guaranteed uniform sample over all 1.61 million rows. If source ordering is structured, the sample may be biased. The resulting UTF-8 text file was `/kaggle/working/assamese_100k.txt`.

The card's declared license and source list are recorded here, not independently verified. Preserve attribution and review source-level terms before redistributing the corpus or derived artifacts.

## Runs and results

All reported runs used the smoke model architecture (`d_model=96`, 3 layers, 4 attention heads; 357,024 parameters), seed 0, batch size 16, learning rate 0.002, and the same text-training path. The tokenizer represents UTF-8 bytes as tokens.

| Run | Training data | Context | Steps | Device | Validation loss, initial to final | Byte perplexity, initial to final | Final training loss | Time |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Initial smoke | 20,000 sentences | 128 bytes | 100 | Not recorded | Not recorded | Not recorded | Not recorded | Not recorded |
| Small pilot | 20,000 sentences | 128 bytes | 1,000 | CPU | 5.453 to 0.962 | 233.4 to 2.616 | 0.923 | 107.7 s |
| Larger pilot | 100,000 sentences | 256 bytes | 3,000 | Tesla T4 (16 GB) | 5.449 to 0.709 | 232.6 to 2.032 | 0.738 | 41.4 s |

Validation uses the last 10% of the concatenated text bytes. The corpus exposes sentence text rather than document/source IDs, so this is not a document- or source-separated evaluation. Perplexity is measured per byte and must not be compared directly with word- or subword-tokenized model scores. The runs also change data volume, context length, step count, and device, so the difference between rows is not a controlled estimate of any single factor.

## Qualitative generation

Both reported checkpoints generated mostly incoherent text for short Assamese prompts. The larger pilot produced plausible-looking script and fragments but not coherent sentences. Samples sometimes ended in the Unicode replacement character (`�`), likely because generation stopped in the middle of a multibyte UTF-8 character.

The loss curves show that optimization and data loading work, but the samples do not demonstrate usable Assamese generation. These checkpoints are exploratory baselines, not quality claims.

## Next steps

- Improve sampling so the pilot represents the full corpus, or document the source-order limitation.
- Keep a held-out evaluation split and avoid claiming source-level generalization without source IDs.
- Evaluate generation quality separately from byte-level loss.
- Collect appropriately licensed, consented informal Assamese data before testing informal-language performance.
