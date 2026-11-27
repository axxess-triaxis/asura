"""lm-evaluation-harness plugin for checkpoints produced by Asura."""

import torch
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from ..device import get_device
from ..model import Asura, AsuraConfig
from ..tokenizer import ByteTokenizer


@register_model("asura")
class AsuraLM(LM):
    """Evaluate an Asura checkpoint with the harness's custom-model interface.

    Asura has no BOS/EOS token. For harness requests that need an initial
    context, byte token 0 (NUL) is used as the documented prefix sentinel.
    """

    def __init__(self, pretrained: str, device: str = "auto", **kwargs):
        super().__init__()
        if kwargs:
            raise ValueError(f"unsupported Asura model arguments: {', '.join(sorted(kwargs))}")
        self._device = get_device() if device == "auto" else torch.device(device)
        checkpoint = torch.load(pretrained, map_location=self._device, weights_only=True)
        if not isinstance(checkpoint, dict) or "model_cfg" not in checkpoint or "model" not in checkpoint:
            raise ValueError("pretrained must be an Asura checkpoint containing model_cfg and model state")
        self.model = Asura(AsuraConfig(**checkpoint["model_cfg"])).to(self._device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.tokenizer = ByteTokenizer()
        self.batch_size = 1

    @property
    def tokenizer_name(self) -> str:
        return "asura-byte-v1"

    @property
    def max_length(self) -> int:
        return self.model.cfg.context_len

    def _score_tokens(self, context: list[int], continuation: list[int]) -> tuple[float, bool]:
        history = context or [0]
        log_likelihood = 0.0
        greedy = True
        for index, token in enumerate(continuation):
            input_ids = (history + continuation[:index])[-self.max_length :]
            tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
            with torch.inference_mode():
                logits, _ = self.model(tensor)
                next_logits = logits[0, -1]
                log_probs = torch.log_softmax(next_logits, dim=-1)
                log_likelihood += log_probs[token].item()
                greedy = greedy and int(next_logits.argmax().item()) == token
        return log_likelihood, greedy

    def loglikelihood(self, requests):
        results = []
        for request in requests:
            context, continuation = request.args
            context_ids = self.tokenizer.encode(context)
            continuation_ids = self.tokenizer.encode(context + continuation)[len(context_ids) :]
            results.append(self._score_tokens(context_ids, continuation_ids))
        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for request in requests:
            token_ids = self.tokenizer.encode(request.args[0])
            results.append(self._score_tokens([0], token_ids)[0])
        return results

    def generate_until(self, requests):
        results = []
        for request in requests:
            context, generation_kwargs = request.args
            unsupported = set(generation_kwargs) - {
                "until",
                "max_gen_toks",
                "temperature",
                "top_k",
            }
            if unsupported:
                raise ValueError(f"unsupported Asura generation arguments: {', '.join(sorted(unsupported))}")
            context_ids = self.tokenizer.encode(context)
            input_ids = context_ids or [0]
            max_new_tokens = int(generation_kwargs.get("max_gen_toks", 128))
            if max_new_tokens < 0:
                raise ValueError("max_gen_toks must be non-negative")
            temperature = generation_kwargs.get("temperature")
            if temperature is None:
                temperature = 0
            top_k = generation_kwargs.get("top_k")
            output = self.model.generate(
                torch.tensor([input_ids], dtype=torch.long, device=self._device),
                max_new_tokens=max_new_tokens,
                temperature=float(temperature),
                top_k=top_k,
            )
            generated = self.tokenizer.decode(output[0, len(input_ids) :].tolist())
            stop_sequences = generation_kwargs.get("until") or []
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            stop_positions = [generated.find(stop) for stop in stop_sequences if stop and stop in generated]
            if stop_positions:
                generated = generated[: min(stop_positions)]
            results.append(generated)
        return results
