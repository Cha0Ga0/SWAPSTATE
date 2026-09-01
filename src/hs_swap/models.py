from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    """A loaded causal LM and tokenizer."""

    model_name: str
    tokenizer: Any
    model: Any
    device: str


def _pick_dtype(dtype: str, device: Optional[str] = None) -> torch.dtype:
    if dtype in ("fp16", "float16"):
        return torch.float16
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    # Keep CPU inference portable; many CPU kernels do not support fp16.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _dtype_load_kwargs(dtype: torch.dtype) -> Dict[str, torch.dtype]:
    """Use the non-deprecated dtype keyword on Transformers 5+."""
    try:
        major = int(transformers.__version__.split(".", 1)[0])
    except (TypeError, ValueError):
        major = 4
    key = "dtype" if major >= 5 else "torch_dtype"
    return {key: dtype}


def load_model_and_tokenizer(
    model_name: str,
    device: Optional[str] = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    additional_special_tokens: Optional[Sequence[str]] = None,
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

    added_token_count = 0
    if additional_special_tokens:
        existing = list(getattr(tok, "additional_special_tokens", None) or [])
        combined = list(dict.fromkeys([*existing, *additional_special_tokens]))
        added_token_count = tok.add_special_tokens(
            {"additional_special_tokens": combined}
        )

    # Decoder-only models should use left padding for batched generation.
    tok.padding_side = "left"
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    torch_dtype = _pick_dtype(dtype, device)
    dtype_kwargs = _dtype_load_kwargs(torch_dtype)

    if device.startswith("cuda") and device != "cuda" and ":" in device:
        # single specific GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **dtype_kwargs,
            device_map=None,
            trust_remote_code=trust_remote_code,
        ).to(device)
    else:
        # allow HF to spread if multiple GPUs; ok on single GPU too
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **dtype_kwargs,
            device_map="auto" if device.startswith("cuda") else None,
            trust_remote_code=trust_remote_code,
        )
        if not device.startswith("cuda"):
            model = model.to(device)

    if added_token_count:
        model.resize_token_embeddings(len(tok))

    model.eval()
    return LoadedModel(model_name=model_name, tokenizer=tok, model=model, device=device)
