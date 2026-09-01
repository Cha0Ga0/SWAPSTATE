"""hs_swap: token-aligned hidden-state swap utilities."""

from .alignment import AlignedInputs, build_aligned_inputs, ensure_state_marker
from .inference import GenerationResult, generate_batch
from .io import append_jsonl, iter_jsonl, read_jsonl
from .models import load_model_and_tokenizer
from .prompting import build_prompt_from_request
from .swapping import (
    CaptureResult,
    InterventionResult,
    capture_hidden_states,
    forward_with_injected_states,
    generate_with_injected_states,
)
