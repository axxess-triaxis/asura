"""Byte-level tokenizer: every UTF-8 byte is a token (vocab = 256).

No training needed, handles any language/text. Upgrade path: train a BPE
tokenizer (see README roadmap, Stage 3) once you want longer effective context.
"""


class ByteTokenizer:
    vocab_size = 256
    newline = ord("\n")

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")
