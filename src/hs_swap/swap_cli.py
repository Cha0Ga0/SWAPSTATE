from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from hs_swap.alignment import build_aligned_inputs, ensure_state_marker
from hs_swap.cli import _has_pending_requests
from hs_swap.io import append_jsonl, iter_jsonl, load_written_custom_ids
from hs_swap.models import load_model_and_tokenizer
from hs_swap.swapping import (
    capture_hidden_states,
    forward_with_injected_states,
    generate_with_injected_states,
    get_input_device,
    select_state_vectors,
)


@dataclass(frozen=True)
class SwapCase:
    custom_id: str
    instructions: List[str]
    question_block: str
    swap_pairs: List[Tuple[int, int]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hs-swap-experiment")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--input", required=True, help="Experiment JSONL")
    parser.add_argument("--output", required=True, help="Result JSONL (append)")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Zero-based decoder layer indices to swap simultaneously",
    )
    parser.add_argument("--state-token", default="[STATE]")
    parser.add_argument("--filler-text", default=" ")
    parser.add_argument("--add-bos", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--pair-batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default=None, help="cuda / cuda:0 / cpu")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")
    if args.pair_batch_size <= 0:
        parser.error("--pair-batch-size must be greater than zero")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    return args


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_options(options: Any) -> str:
    if isinstance(options, dict):
        return " ".join(f"{key}. {value}" for key, value in options.items())
    if isinstance(options, list):
        return " ".join(
            f"{chr(ord('A') + index)}. {value}"
            for index, value in enumerate(options)
        )
    raise ValueError("options must be an object or an array")


def _validated_instruction_list(value: Any, label: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    if not all(isinstance(instruction, str) for instruction in value):
        raise ValueError(f"every instruction in {label} must be a string")
    return list(value)


def _extract_case(record: Dict[str, Any], state_token: str) -> SwapCase:
    custom_id = str(record.get("custom_id", ""))
    if not custom_id:
        raise ValueError("each experiment record requires a non-empty custom_id")

    instruction_groups = record.get("instruction_groups")
    if instruction_groups is not None:
        if not isinstance(instruction_groups, list) or len(instruction_groups) < 2:
            raise ValueError("instruction_groups must contain at least two arrays")
        instructions: List[str] = []
        group_ids: List[int] = []
        for group_index, group in enumerate(instruction_groups):
            validated = _validated_instruction_list(
                group,
                f"instruction_groups[{group_index}]",
            )
            instructions.extend(validated)
            group_ids.extend([group_index] * len(validated))
    else:
        instructions = _validated_instruction_list(
            record.get("instructions"),
            "instructions",
        )
        if len(instructions) < 2:
            raise ValueError("instructions must contain at least two strings")
        group_ids = list(range(len(instructions)))

    configured_pairs = record.get("swap_pairs")
    if configured_pairs is None:
        swap_pairs = [
            (source_index, target_index)
            for target_index in range(len(instructions))
            for source_index in range(len(instructions))
            if group_ids[source_index] != group_ids[target_index]
        ]
    else:
        if not isinstance(configured_pairs, list) or not configured_pairs:
            raise ValueError("swap_pairs must be a non-empty array")
        swap_pairs = []
        for pair in configured_pairs:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(type(index) is int for index in pair)
            ):
                raise ValueError("each swap pair must be [source_index, target_index]")
            swap_pairs.append((pair[0], pair[1]))

    if len(set(swap_pairs)) != len(swap_pairs):
        raise ValueError("swap_pairs must be unique")
    for source_index, target_index in swap_pairs:
        if source_index == target_index:
            raise ValueError("a swap source and target must differ")
        if not (
            0 <= source_index < len(instructions)
            and 0 <= target_index < len(instructions)
        ):
            raise ValueError("swap pair index is outside the instruction array")

    question_block = record.get("question_block")
    if question_block is None:
        question = record.get("question")
        if not isinstance(question, str):
            raise ValueError("record requires question_block or question")
        parts = [question]
        if "options" in record:
            parts.append(_format_options(record["options"]))
        question_block = "\n".join(parts)
    if not isinstance(question_block, str):
        raise ValueError("question_block must be a string")

    return SwapCase(
        custom_id=custom_id,
        instructions=instructions,
        question_block=ensure_state_marker(question_block, state_token),
        swap_pairs=swap_pairs,
    )


def _generation_kwargs(args: argparse.Namespace, tokenizer: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.do_sample:
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
    return kwargs


def _decode_continuations(
    tokenizer: Any,
    output_ids: torch.LongTensor,
    prompt_length: int,
) -> List[str]:
    return [
        text.strip()
        for text in tokenizer.batch_decode(
            output_ids[:, prompt_length:],
            skip_special_tokens=True,
        )
    ]


def _run_case(
    record: Dict[str, Any],
    loaded: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    case = _extract_case(record, args.state_token)
    aligned = build_aligned_inputs(
        loaded.tokenizer,
        case.instructions,
        case.question_block,
        state_token=args.state_token,
        filler_text=args.filler_text,
        add_bos=args.add_bos,
    )
    inputs = aligned.as_model_inputs(get_input_device(loaded.model))
    prompt_length = int(inputs["input_ids"].shape[1])

    baseline = capture_hidden_states(
        loaded.model,
        inputs,
        aligned.state_position,
        args.layers,
    )
    generation_kwargs = _generation_kwargs(args, loaded.tokenizer)
    with torch.no_grad():
        baseline_ids = loaded.model.generate(
            **inputs,
            **generation_kwargs,
        ).detach().cpu()
    baseline_texts = _decode_continuations(
        loaded.tokenizer,
        baseline_ids,
        prompt_length,
    )
    baseline_answer_ids = baseline.logits.argmax(dim=-1).tolist()
    baselines = [
        {
            "instruction_index": index,
            "instruction": instruction,
            "next_token_id": baseline_answer_ids[index],
            "completion": baseline_texts[index],
        }
        for index, instruction in enumerate(case.instructions)
    ]

    swaps: List[Dict[str, Any]] = []
    for start in range(0, len(case.swap_pairs), args.pair_batch_size):
        pair_chunk = case.swap_pairs[start : start + args.pair_batch_size]
        source_indices = [source for source, _ in pair_chunk]
        target_indices = [target for _, target in pair_chunk]

        device_target_indices = torch.tensor(
            target_indices,
            dtype=torch.long,
            device=inputs["input_ids"].device,
        )
        pair_inputs = {
            key: value.index_select(0, device_target_indices)
            for key, value in inputs.items()
        }
        injected_states = select_state_vectors(
            baseline.state_vectors,
            source_indices,
        )
        target_baseline_logits = baseline.logits.index_select(
            0,
            torch.tensor(target_indices, dtype=torch.long),
        )

        intervened = forward_with_injected_states(
            loaded.model,
            pair_inputs,
            aligned.state_position,
            injected_states,
            target_baseline_logits,
        )
        swapped_ids = generate_with_injected_states(
            loaded.model,
            pair_inputs,
            aligned.state_position,
            injected_states,
            **generation_kwargs,
        )
        swapped_texts = _decode_continuations(
            loaded.tokenizer,
            swapped_ids,
            prompt_length,
        )
        swapped_answer_ids = intervened.logits.argmax(dim=-1).tolist()

        for pair_index, (source_index, target_index) in enumerate(pair_chunk):
            swaps.append(
                {
                    "source_instruction_index": source_index,
                    "target_instruction_index": target_index,
                    "next_token_id": swapped_answer_ids[pair_index],
                    "completion": swapped_texts[pair_index],
                    "cosine_distance": float(
                        intervened.cosine_distances[pair_index]
                    ),
                    "kl_divergence": float(
                        intervened.kl_divergences[pair_index]
                    ),
                }
            )

    return {
        "custom_id": case.custom_id,
        "status": "ok",
        "layers": list(args.layers),
        "state_token": args.state_token,
        "state_position": aligned.state_position,
        "instruction_token_lengths": aligned.instruction_token_lengths,
        "filler_counts": aligned.filler_counts,
        "baselines": baselines,
        "swaps": swaps,
        "error": None,
    }


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    written = load_written_custom_ids(args.output) if args.resume else set()
    if args.resume and not _has_pending_requests(args.input, written):
        return

    loaded = load_model_and_tokenizer(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        additional_special_tokens=[args.state_token],
    )

    scheduled = set(written)
    for record in tqdm(iter_jsonl(args.input), desc="swap cases"):
        custom_id = str(record.get("custom_id", ""))
        if args.resume and custom_id in scheduled:
            continue
        if args.resume:
            scheduled.add(custom_id)
        try:
            result = _run_case(record, loaded, args)
        except Exception as exc:
            if args.fail_fast:
                raise
            result = {
                "custom_id": custom_id,
                "status": "error",
                "layers": list(args.layers),
                "state_token": args.state_token,
                "error": f"{type(exc).__name__}: {exc}",
            }
        append_jsonl(args.output, result)


if __name__ == "__main__":
    main()
