from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class BatchStopOnStrings(StoppingCriteria):
    """Stop generation when *all* samples contain any stop string.

    This avoids the common failure mode where one sample hits a stop string and
    truncates the whole batch early.
    """

    def __init__(self, stop_strings: Sequence[str], tokenizer: Any):
        super().__init__()
        self.stop_strings = [s for s in stop_strings if s]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_strings:
            return False
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        done = []
        for t in texts:
            done.append(any(s in t for s in self.stop_strings))
        return all(done)


def strip_prompt(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):]
    return full_text


def postprocess_stop_strings(text: str, stop_strings: Sequence[str]) -> str:
    out = text
    for s in stop_strings:
        if not s:
            continue
        idx = out.find(s)
        if idx != -1:
            out = out[:idx]
    return out


def try_parse_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Best-effort parse of the first JSON object found in text."""
    import json

    # fast path: entire string is JSON
    try:
        return json.loads(text), None
    except Exception:
        pass

    # fallback: find the first JSON object substring
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None, "no_json_found"
    chunk = m.group(0)
    try:
        return json.loads(chunk), None
    except Exception as e:
        return None, f"json_parse_error: {e}"



def generate_batch(
    loaded,
    prompts: List[str],
    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 0.3,
    top_p: Optional[float] = None,
    stop_strings: Optional[List[str]] = None,
) -> List[str]:
    """Generate continuations for a list of prompts."""
    tok = loaded.tokenizer
    model = loaded.model

    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    # eos handling: support multiple eos ids if tokenizer provides them
    eos_ids = []
    if tok.eos_token_id is not None:
        eos_ids.append(tok.eos_token_id)
    # some instruct models have eot_id in additional_special_tokens
    if hasattr(tok, "additional_special_tokens_ids"):
        for i in getattr(tok, "additional_special_tokens_ids") or []:
            if i not in eos_ids:
                eos_ids.append(i)

    stopping = StoppingCriteriaList()
    if stop_strings:
        stopping.append(BatchStopOnStrings(stop_strings=stop_strings, tokenizer=tok))

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        eos_token_id=eos_ids if eos_ids else None,
        pad_token_id=tok.pad_token_id,
        stopping_criteria=stopping,
    )
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        out_ids = model.generate(**enc, **{k: v for k, v in gen_kwargs.items() if v is not None})

    texts = tok.batch_decode(out_ids, skip_special_tokens=False)
    return texts
