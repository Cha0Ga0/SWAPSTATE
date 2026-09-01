from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch


@dataclass(frozen=True)
class AlignedInputs:
    """Token-aligned prompts sharing one intervention-marker position."""

    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    state_position: int
    state_token_id: int
    filler_counts: List[int]
    instruction_token_lengths: List[int]

    def as_model_inputs(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids.to(device),
            "attention_mask": self.attention_mask.to(device),
        }


def _encode_without_special_tokens(tokenizer: Any, text: str) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        raise ValueError("expected one text input, received batched token ids")
    return [int(token_id) for token_id in encoded]


def get_single_token_id(tokenizer: Any, text: str, *, label: str) -> int:
    token_ids = _encode_without_special_tokens(tokenizer, text)
    if len(token_ids) != 1:
        raise ValueError(
            f"{label} must encode to exactly one token; got {len(token_ids)} tokens"
        )
    return token_ids[0]


def ensure_state_marker(question_block: str, state_token: str) -> str:
    """Append the state marker when the input question does not contain it."""
    occurrences = question_block.count(state_token)
    if occurrences > 1:
        raise ValueError("question_block must contain at most one state marker")
    if occurrences == 1:
        return question_block
    return f"{question_block.rstrip()}\n{state_token}"


def build_aligned_inputs(
    tokenizer: Any,
    instructions: Sequence[str],
    question_block: str,
    *,
    state_token: str = "[STATE]",
    filler_text: str = " ",
    add_bos: bool = False,
) -> AlignedInputs:
    """Align a common question suffix across instruction variants.

    Each sequence is constructed directly in token space as::

        [BOS?] instruction NEWLINE filler... NEWLINE question-with-[STATE]

    The filler is an attended token, not attention-mask padding. This preserves a
    shared absolute position for every question token and for ``state_token``.
    """
    if len(instructions) < 2:
        raise ValueError("at least two instructions are required for state swapping")

    state_token_id = get_single_token_id(
        tokenizer,
        state_token,
        label="state_token",
    )
    filler_token_id = get_single_token_id(
        tokenizer,
        filler_text,
        label="filler_text",
    )
    newline_ids = _encode_without_special_tokens(tokenizer, "\n")
    if not newline_ids:
        raise ValueError("newline must encode to at least one token")

    question_ids = _encode_without_special_tokens(tokenizer, question_block)
    marker_offsets = [
        index for index, token_id in enumerate(question_ids) if token_id == state_token_id
    ]
    if len(marker_offsets) != 1:
        raise ValueError(
            "question_block must contain exactly one occurrence of state_token"
        )
    marker_offset = marker_offsets[0]

    instruction_ids = [
        _encode_without_special_tokens(tokenizer, instruction)
        for instruction in instructions
    ]
    if any(state_token_id in token_ids for token_ids in instruction_ids):
        raise ValueError("instructions must not contain the state marker token")

    base_lengths = [
        len(token_ids) + 2 * len(newline_ids) for token_ids in instruction_ids
    ]
    target_instruction_length = max(base_lengths)
    filler_counts = [
        target_instruction_length - base_length for base_length in base_lengths
    ]

    prefix: List[int] = []
    if add_bos:
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if bos_token_id is None:
            raise ValueError("add_bos=True but tokenizer has no bos_token_id")
        prefix = [int(bos_token_id)]

    sequences: List[List[int]] = []
    for token_ids, filler_count in zip(instruction_ids, filler_counts):
        sequence = list(prefix)
        sequence.extend(token_ids)
        sequence.extend(newline_ids)
        sequence.extend([filler_token_id] * filler_count)
        sequence.extend(newline_ids)
        sequence.extend(question_ids)
        sequences.append(sequence)

    sequence_length = len(sequences[0])
    if any(len(sequence) != sequence_length for sequence in sequences):
        raise RuntimeError("token alignment failed: sequence lengths differ")

    state_position = (
        len(prefix) + target_instruction_length + marker_offset
    )
    input_ids = torch.tensor(sequences, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    if not torch.all(input_ids[:, state_position] == state_token_id):
        raise RuntimeError("token alignment failed: state marker positions differ")

    return AlignedInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        state_position=state_position,
        state_token_id=state_token_id,
        filler_counts=filler_counts,
        instruction_token_lengths=[len(ids) for ids in instruction_ids],
    )
