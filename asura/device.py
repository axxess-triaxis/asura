"""Pick the best available device: NVIDIA GPU (Colab/Kaggle/cloud) > Apple GPU > CPU.

Override with the environment variable ASURA_DEVICE=cpu (or cuda, mps).
"""
import os
import torch


def get_device() -> torch.device:
    want = os.environ.get("ASURA_DEVICE")
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe(dev: torch.device) -> str:
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        return f"cuda ({p.name}, {p.total_memory / 1e9:.0f} GB)"
    return dev.type
