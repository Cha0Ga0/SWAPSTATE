from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


@dataclass(frozen=True)
class GenerationResult:
    """Decoded full output and the token-exact generated continuation."""

    raw_output: str
    completion: str


class BatchStopOnStrings(StoppingCriteria):
    """Stop generation when all samples contain a stop string in generated text."""

    def __init__(
        self,
        stop_strings: Sequence[str],
        tokenizer: Any,
        prompt_length: int = 0,
    ):
        super().__init__()
        self.stop_strings = [s for s in stop_strings if s]
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        if not self.stop_strings:
            return False
        generated_ids = input_ids[:, self.prompt_length :]
        texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )
        return all(any(s in text for s in self.stop_strings) for text in texts)


def strip_prompt(full_text: str, prompt: str) -> str:
    """Legacy text-prefix stripping helper.

    Generation uses token boundaries instead; this remains for API compatibility.
    """
    if full_text.startswith(prompt):
        return full_text[len(prompt) :]
    return full_text


def postprocess_stop_strings(text: str, stop_strings: Sequence[str]) -> str:
    out = text
    for stop_string in stop_strings:
        if not stop_string:
            continue
        idx = out.find(stop_string)
        if idx != -1:
            out = out[:idx]
    return out


def try_parse_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Best-effort parse of the first JSON object found in text."""
    import json

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if not isinstance(parsed, dict):
            return None, "json_value_is_not_an_object"
        return parsed, None

    decoder = json.JSONDecoder()
    last_error: Optional[json.JSONDecodeError] = None
    found_opening_brace = False
    for match in re.finditer(r"\{", text):
        found_opening_brace = True
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed, None

    if not found_opening_brace:
        return None, "no_json_found"
    if last_error is not None:
        return None, f"json_parse_error: {last_error}"
    return None, "json_object_not_found"


def _collect_eos_token_ids(tokenizer: Any, model: Any) -> List[int]:
    """Collect only configured EOS ids, not every additional special token."""
    eos_ids: List[int] = []

    def add(value: Any) -> None:
        values = value if isinstance(value, (list, tuple)) else [value]
        for token_id in values:
            if token_id is None:
                continue
            normalized = int(token_id)
            if normalized not in eos_ids:
                eos_ids.append(normalized)

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        add(getattr(generation_config, "eos_token_id", None))
    add(getattr(tokenizer, "eos_token_id", None))
    return eos_ids


def generate_batch(
    loaded: Any,
    prompts: List[str],
    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 0.3,
    top_p: Optional[float] = None,
    stop_strings: Optional[List[str]] = None,
) -> List[GenerationResult]:
    """Generate and decode full outputs plus token-exact continuations."""
    tokenizer = loaded.tokenizer
    model = loaded.model

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    eos_ids = _collect_eos_token_ids(tokenizer, model)

    stopping = StoppingCriteriaList()
    if stop_strings:
        stopping.append(
            BatchStopOnStrings(
                stop_strings=stop_strings,
                tokenizer=tokenizer,
                prompt_length=prompt_length,
            )
        )

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "eos_token_id": eos_ids if eos_ids else None,
        "pad_token_id": tokenizer.pad_token_id,
        "stopping_criteria": stopping,
    }
    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            **{
                key: value
                for key, value in generation_kwargs.items()
                if value is not None
            },
        )

    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        raw_sequences = list(output_ids)
    else:
        left_padding = prompt_length - attention_mask.sum(dim=1)
        raw_sequences = [
            sequence[int(padding.item()) :]
            for sequence, padding in zip(output_ids, left_padding)
        ]
    raw_texts = tokenizer.batch_decode(
        raw_sequences,
        skip_special_tokens=False,
    )
    completion_texts = tokenizer.batch_decode(
        output_ids[:, prompt_length:],
        skip_special_tokens=False,
    )
    return [
        GenerationResult(raw_output=raw, completion=completion)
        for raw, completion in zip(raw_texts, completion_texts)
    ]
