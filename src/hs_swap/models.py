from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    """A loaded causal LM and tokenizer."""

    model_name: str
    tokenizer: any
    model: any
    device: str


def _pick_dtype(dtype: str) -> torch.dtype:
    if dtype in ("fp16", "float16"):
        return torch.float16
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    # auto
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer(
    model_name: str,
    device: Optional[str] = None,
    dtype: str = "auto",
    trust_remote_code: bool = True,
) -> LoadedModel:
    """Load a HuggingFace decoder-only model for local generation.

    Notes:
    - For decoder-only models, we set left padding to avoid the right-padding warning.
    - Uses device_map='auto' for multi-GPU; otherwise places on a single device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    # Decoder-only models should use left padding for batched generation.
    tok.padding_side = "left"
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    torch_dtype = _pick_dtype(dtype)

    if device.startswith("cuda") and device != "cuda" and ":" in device:
        # single specific GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=trust_remote_code,
        ).to(device)
    else:
        # allow HF to spread if multiple GPUs; ok on single GPU too
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device.startswith("cuda") else None,
            trust_remote_code=trust_remote_code,
        )
        if not device.startswith("cuda"):
            model = model.to(device)

    model.eval()
    return LoadedModel(model_name=model_name, tokenizer=tok, model=model, device=device)
