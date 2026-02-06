#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Batch local inference for API-style JSONL requests.

Input format (recommended): OpenAI-style JSONL (one request per line)

Example line:
{
  "custom_id": "Mercury_7175875-1",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ]
  }
}

The script writes one JSON object per input line to OUTPUT_JSONL:
{
  "custom_id": "...",
  "raw_output": "..."...
}
"""

import argparse
import json
from typing import Any, Dict, List

from tqdm import tqdm

from hs_swap.io import append_jsonl, iter_jsonl, load_written_custom_ids
from hs_swap.models import load_model_and_tokenizer
from hs_swap.prompting import build_prompt_from_request
from hs_swap.inference import generate_batch, strip_prompt, postprocess_stop_strings, try_parse_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model name")
    p.add_argument("--input", required=True, help="Input JSONL")
    p.add_argument("--output", required=True, help="Output JSONL (append)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--stop", nargs="*", default=[], help="Optional stop strings")
    p.add_argument("--device", default=None, help="e.g. cuda:0 / cuda / cpu")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--no-trust-remote-code", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip custom_id already in output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trust_remote_code = True
    if args.no_trust_remote_code:
        trust_remote_code = False
    if args.trust_remote_code:
        trust_remote_code = True

    loaded = load_model_and_tokenizer(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=trust_remote_code,
    )

    written = load_written_custom_ids(args.output) if args.resume else set()

    buffer_reqs: List[Dict[str, Any]] = []
    buffer_prompts: List[str] = []

    def flush() -> None:
        if not buffer_reqs:
            return
        raw_texts = generate_batch(
            loaded,
            prompts=buffer_prompts,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_strings=args.stop,
        )
        for req, prompt, raw in zip(buffer_reqs, buffer_prompts, raw_texts):
            cid = str(req.get("custom_id", ""))
            continuation = strip_prompt(raw, prompt)
            continuation = postprocess_stop_strings(continuation, args.stop)
            json_out, err = try_parse_json(continuation)

            record: Dict[str, Any] = {
                "custom_id": cid,
                "raw_output": continuation,
            }
            if json_out is not None:
                record["json_output"] = json_out
            if err is not None:
                record["parse_error"] = err

            append_jsonl(args.output, record)

        buffer_reqs.clear()
        buffer_prompts.clear()

    # stream read + batched flush
    for req in tqdm(iter_jsonl(args.input), desc="requests"):
        cid = str(req.get("custom_id", ""))
        if args.resume and cid in written:
            continue
        prompt = build_prompt_from_request(loaded.tokenizer, req)

        buffer_reqs.append(req)
        buffer_prompts.append(prompt)

        if len(buffer_reqs) >= args.batch_size:
            flush()

    flush()


if __name__ == "__main__":
    main()
