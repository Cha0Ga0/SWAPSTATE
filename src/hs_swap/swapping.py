from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class CaptureResult:
    """Last-token logits and captured marker states, stored on CPU."""

    logits: torch.Tensor
    state_vectors: Dict[int, torch.Tensor]


@dataclass(frozen=True)
class InterventionResult:
    """Intervened last-token logits and distances from baseline logits."""

    logits: torch.Tensor
    cosine_distances: torch.Tensor
    kl_divergences: torch.Tensor


def get_input_device(model: nn.Module) -> torch.device:
    """Return the embedding device, including for device-mapped models."""
    embedding = model.get_input_embeddings()
    return embedding.weight.device


def _resolve_attribute(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part)
    return value


def resolve_decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    """Resolve decoder blocks for common HuggingFace causal-LM architectures."""
    candidates = (
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.decoder.layers",
        "base_model.model.model.layers",
        "base_model.model.layers",
    )
    for path in candidates:
        try:
            layers = _resolve_attribute(model, path)
        except AttributeError:
            continue
        if isinstance(layers, (nn.ModuleList, list, tuple)) and layers:
            return layers
    raise ValueError(
        "could not locate decoder layers; supported paths include "
        "model.layers, transformer.h, and gpt_neox.layers"
    )


def normalize_layer_indices(
    layer_indices: Sequence[int],
    layer_count: int,
) -> List[int]:
    if not layer_indices:
        raise ValueError("at least one layer index is required")
    normalized = [int(index) for index in layer_indices]
    if len(set(normalized)) != len(normalized):
        raise ValueError("layer indices must be unique")
    invalid = [index for index in normalized if index < 0 or index >= layer_count]
    if invalid:
        raise ValueError(
            f"layer indices out of range for {layer_count} layers: {invalid}"
        )
    return normalized


def _hidden_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(
        "decoder layer output must be a tensor or a tuple/list whose first item "
        "is a tensor"
    )


def _replace_hidden_in_output(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError("unsupported decoder layer output type")


def _last_token_logits(outputs: Any, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    logits = outputs.logits
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        return logits[:, -1, :]
    positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    masked_positions = positions.masked_fill(~attention_mask.to(torch.bool), -1)
    last_positions = masked_positions.max(dim=1).values
    if torch.any(last_positions < 0):
        raise ValueError("attention_mask contains an empty sequence")
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, last_positions, :]


def capture_hidden_states(
    model: nn.Module,
    inputs: Mapping[str, torch.Tensor],
    state_position: int,
    layer_indices: Sequence[int],
) -> CaptureResult:
    """Run one forward pass and capture decoder-layer outputs at the marker."""
    layers = resolve_decoder_layers(model)
    selected = normalize_layer_indices(layer_indices, len(layers))
    prompt_length = int(inputs["input_ids"].shape[1])
    if state_position < 0 or state_position >= prompt_length:
        raise ValueError("state_position is outside the input sequence")

    captured: Dict[int, torch.Tensor] = {}
    handles: List[Any] = []

    def make_hook(layer_index: int):
        def hook(module: nn.Module, hook_inputs: Tuple[Any, ...], output: Any):
            del module, hook_inputs
            if layer_index in captured:
                return output
            hidden = _hidden_from_output(output)
            if hidden.ndim != 3 or state_position >= hidden.shape[1]:
                raise RuntimeError(
                    f"layer {layer_index} did not expose marker position "
                    f"{state_position}"
                )
            captured[layer_index] = hidden[:, state_position, :].detach().cpu()
            return output

        return hook

    try:
        for layer_index in selected:
            handles.append(
                layers[layer_index].register_forward_hook(make_hook(layer_index))
            )
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = [index for index in selected if index not in captured]
    if missing:
        raise RuntimeError(f"failed to capture hidden states at layers {missing}")

    logits = _last_token_logits(outputs, inputs).detach().float().cpu()
    return CaptureResult(logits=logits, state_vectors=captured)


def select_state_vectors(
    state_vectors: Mapping[int, torch.Tensor],
    source_indices: Sequence[int],
) -> Dict[int, torch.Tensor]:
    indices = torch.tensor(source_indices, dtype=torch.long)
    return {
        layer_index: vectors.index_select(0, indices)
        for layer_index, vectors in state_vectors.items()
    }


@contextmanager
def _injection_hooks(
    model: nn.Module,
    inputs: Mapping[str, torch.Tensor],
    state_position: int,
    layer_states: Mapping[int, torch.Tensor],
) -> Iterator[Dict[int, int]]:
    layers = resolve_decoder_layers(model)
    selected = normalize_layer_indices(list(layer_states), len(layers))
    batch_size, prompt_length = inputs["input_ids"].shape
    if state_position < 0 or state_position >= prompt_length:
        raise ValueError("state_position is outside the input sequence")

    for layer_index in selected:
        vectors = layer_states[layer_index]
        if vectors.ndim != 2 or vectors.shape[0] != batch_size:
            raise ValueError(
                f"layer {layer_index} replacement must have shape "
                f"({batch_size}, hidden_size)"
            )

    injection_counts = {layer_index: 0 for layer_index in selected}
    handles: List[Any] = []

    def make_hook(layer_index: int, vectors: torch.Tensor):
        def hook(module: nn.Module, hook_inputs: Tuple[Any, ...], output: Any):
            del module, hook_inputs
            if injection_counts[layer_index] > 0:
                return output
            hidden = _hidden_from_output(output)
            if hidden.ndim != 3 or state_position >= hidden.shape[1]:
                raise RuntimeError(
                    f"first layer-{layer_index} forward did not include marker "
                    f"position {state_position}"
                )
            if hidden.shape[0] != vectors.shape[0] or hidden.shape[2] != vectors.shape[1]:
                raise ValueError(
                    f"layer {layer_index} replacement shape {tuple(vectors.shape)} "
                    f"does not match hidden shape {tuple(hidden.shape)}"
                )
            replacement = vectors.to(device=hidden.device, dtype=hidden.dtype)
            modified = hidden.clone()
            modified[:, state_position, :] = replacement
            injection_counts[layer_index] += 1
            return _replace_hidden_in_output(output, modified)

        return hook

    try:
        for layer_index in selected:
            handles.append(
                layers[layer_index].register_forward_hook(
                    make_hook(layer_index, layer_states[layer_index])
                )
            )
        yield injection_counts
    finally:
        for handle in handles:
            handle.remove()


def _verify_injections(injection_counts: Mapping[int, int]) -> None:
    missing = [index for index, count in injection_counts.items() if count != 1]
    if missing:
        raise RuntimeError(f"hidden-state injection did not run once at layers {missing}")


def compute_logit_metrics(
    baseline_logits: torch.Tensor,
    intervened_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    baseline = baseline_logits.detach().float()
    intervened = intervened_logits.detach().float()
    if baseline.shape != intervened.shape:
        raise ValueError("baseline and intervened logits must have identical shapes")
    cosine_distances = (
        1.0
        - F.cosine_similarity(
            baseline,
            intervened,
            dim=-1,
        )
    ).clamp_min(0.0)
    kl_divergences = (
        F.kl_div(
            F.log_softmax(intervened, dim=-1),
            F.softmax(baseline, dim=-1),
            reduction="none",
        ).sum(dim=-1)
    ).clamp_min(0.0)
    return cosine_distances.cpu(), kl_divergences.cpu()


def forward_with_injected_states(
    model: nn.Module,
    inputs: Mapping[str, torch.Tensor],
    state_position: int,
    layer_states: Mapping[int, torch.Tensor],
    baseline_logits: torch.Tensor,
) -> InterventionResult:
    """Run one no-cache forward pass with marker states replaced."""
    with _injection_hooks(model, inputs, state_position, layer_states) as counts:
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    _verify_injections(counts)

    logits = _last_token_logits(outputs, inputs).detach().float().cpu()
    cosine_distances, kl_divergences = compute_logit_metrics(
        baseline_logits,
        logits,
    )
    return InterventionResult(
        logits=logits,
        cosine_distances=cosine_distances,
        kl_divergences=kl_divergences,
    )


def generate_with_injected_states(
    model: nn.Module,
    inputs: Mapping[str, torch.Tensor],
    state_position: int,
    layer_states: Mapping[int, torch.Tensor],
    **generation_kwargs: Any,
) -> torch.LongTensor:
    """Generate with each configured marker state injected exactly once."""
    with _injection_hooks(model, inputs, state_position, layer_states) as counts:
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
    _verify_injections(counts)
    return output_ids.detach().cpu()
