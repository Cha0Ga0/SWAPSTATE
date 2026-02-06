"""hs_swap: token-aligned hidden-state swap utilities."""

from .io import read_jsonl, iter_jsonl, append_jsonl
from .models import load_model_and_tokenizer
from .prompting import build_prompt_from_request
from .inference import generate_batch
