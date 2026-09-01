#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Batch local inference for API-style JSONL requests."""

import argparse
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from hs_swap.inference import (
    generate_batch,
    postprocess_stop_strings,
    try_parse_json,
)
from hs_swap.io import append_jsonl, iter_jsonl, load_written_custom_ids
from hs_swap.models import load_model_and_tokenizer
from hs_swap.prompting import build_prompt_from_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hs-swap-inference")
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL (append)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--stop",
        "--stop-strings",
        dest="stop_strings",
        nargs="*",
        default=[],
        help="Optional stop strings",
    )
    parser.add_argument("--device", default=None, help="e.g. cuda:0 / cuda / cpu")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
    )
    trust_group = parser.add_mutually_exclusive_group()
    trust_group.add_argument("--trust-remote-code", action="store_true")
    trust_group.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(trust_remote_code=False)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip custom_id already in output",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero")
    return args


def _has_pending_requests(input_path: str, written: set[str]) -> bool:
    scheduled = set(written)
    for request in iter_jsonl(input_path):
        custom_id = str(request.get("custom_id", ""))
        if custom_id not in scheduled:
            return True
    return False


def main() -> None:
    args = parse_args()

    written = load_written_custom_ids(args.output) if args.resume else set()
    scheduled = set(written)
    if args.resume and not _has_pending_requests(args.input, written):
        return

    loaded = load_model_and_tokenizer(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    buffer_reqs: List[Dict[str, Any]] = []
    buffer_prompts: List[str] = []

    def flush() -> None:
        if not buffer_reqs:
            return
        results = generate_batch(
            loaded,
            prompts=buffer_prompts,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_strings=args.stop_strings,
        )
        if len(results) != len(buffer_reqs):
            raise RuntimeError(
                "model generation returned a different number of results than requests"
            )
        for req, result in zip(buffer_reqs, results):
            custom_id = str(req.get("custom_id", ""))
            completion = postprocess_stop_strings(
                result.completion,
                args.stop_strings,
            )
            json_output, parse_error = try_parse_json(completion)
            record: Dict[str, Any] = {
                "custom_id": custom_id,
                "raw_output": result.raw_output,
                "completion": completion,
                "json_output": json_output,
                "parse_error": parse_error,
            }
            append_jsonl(args.output, record)

        buffer_reqs.clear()
        buffer_prompts.clear()

    for req in tqdm(iter_jsonl(args.input), desc="requests"):
        custom_id = str(req.get("custom_id", ""))
        if args.resume and custom_id in scheduled:
            continue
        if args.resume:
            scheduled.add(custom_id)

        prompt = build_prompt_from_request(loaded.tokenizer, req)
        buffer_reqs.append(req)
        buffer_prompts.append(prompt)

        if len(buffer_reqs) >= args.batch_size:
            flush()

    flush()


if __name__ == "__main__":
    main()
