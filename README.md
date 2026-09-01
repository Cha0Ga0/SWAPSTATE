# StateSwap: Probing Support–Elimination Hidden States in Multiple-Choice Questions

<p align="right">English | <a href="README_zh.md">简体中文</a></p>

> Code for the EMNLP 2026 paper **“StateSwap: Probing Support–Elimination
> Hidden States in Multiple-Choice Questions.”**

## Abstract

Large language models often answer the same multiple-choice question
inconsistently when it is posed under support-oriented and elimination-oriented
framings. We investigate whether these discrepancies arise from different
internal representations induced by the two framings. We introduce a
dual-framing protocol with minimally varied prompts that use either support- or
elimination-oriented framing while keeping the evaluation target fixed. To probe
the internal computation, we append an untrained special token, `[STATE]`, and
treat its residual-stream activation as an intervention interface. Across both
models, the two framings induce separable `[STATE]` activations concentrated in
intermediate layers. Swapping these activations between paired prompts
systematically changes predictions and improves cross-framing agreement,
providing intervention-based evidence that the activations are behaviorally
relevant. Beyond instance-level substitution, mean-difference steering
directions derived from the dual-framing contrast exhibit more bounded
layer-wise responses than matched contrastive activation addition directions
under the evaluated protocol.

## Implementation status

SWAPSTATE provides reproducible local inference and token-aligned hidden-state
interventions for HuggingFace decoder-only language models.

Baseline inference, marker alignment, layer-output capture, pairwise hidden-state
injection, intervened generation, and logit metrics are implemented and tested.
The full layers 10–19 intervention path has also been validated with
Qwen2.5-7B-Instruct on a single 40 GB NVIDIA A100.

## Features

- Batched local inference from OpenAI-style JSONL requests.
- Streaming output and resume by `custom_id`.
- Prompt-safe stop strings and token-exact completion decoding.
- A configurable single-token `[STATE]` intervention marker.
- Token-space alignment of a common question suffix across instruction variants.
- Capture and simultaneous injection at arbitrary zero-based decoder layers.
- Every ordered `source instruction → target instruction` swap for N variants.
- Baseline/swapped generation, next-token IDs, cosine distance, and KL divergence.
- Decoder-layer discovery for LLaMA/Qwen/Mistral/Gemma-style, GPT-2, GPT-NeoX,
  and common encoder-decoder paths.

## Installation

Python 3.9 or newer is required. A CUDA GPU is strongly recommended for 7B+
models; the code also supports CPU execution for small-model testing.

```bash
conda create --name stateswap python=3.10
conda activate stateswap
pip install -r requirements.txt
pip install -e .
```

This installs two commands:

- `hs-swap-inference` for ordinary batched generation;
- `hs-swap-experiment` for hidden-state interventions.

Compatibility wrappers remain available as `python scripts/run_inference.py`
and `python scripts/run_swap.py`.

## Hidden-state swap protocol

For every experiment case:

1. `[STATE]` is registered as one additional special token before model loading;
   the model embedding table is resized when needed.
2. Every instruction is tokenized without model-added special tokens.
3. Attended single-token fillers are inserted between each instruction and the
   shared question so that the entire question suffix and `[STATE]` have the
   same absolute positions in every variant.
4. A baseline forward pass captures each selected decoder block's output at the
   marker position.
5. For each ordered instruction pair, source vectors replace target vectors at
   all selected layers.
6. Injection happens once during the full-prompt forward/prefill. Hooks are
   always removed with `finally`, including on errors.
7. The intervened next-token distribution is compared with the target baseline,
   and generation is repeated under the same intervention.

Layer indices are zero-based. Passing several layers swaps all of them in the
same forward pass, matching the notebook experiment.

### Swap input

Each JSONL line contains one common question and at least two instructions:

```json
{"custom_id":"capital-france","instructions":["Answer with the correct option.","Reason briefly, then answer with the correct option."],"question":"What is the capital of France?","options":{"A":"Paris","B":"Rome","C":"Berlin"}}
```

`options` may be an object or array. Alternatively, provide a fully formatted
`question_block`. The CLI appends `[STATE]` when it is absent and rejects more
than one marker. See `examples/swap_cases.jsonl`.

For the notebook's 2×2 design, use `instruction_groups`:

```json
{"custom_id":"case-2x2","instruction_groups":[["wrong-1","wrong-2"],["correct-1","correct-2"]],"question":"..."}
```

This generates swaps in both directions across groups but excludes within-group
swaps. See `examples/swap_group_case.jsonl`. An explicit `swap_pairs` array of
`[source_index, target_index]` entries overrides either default.

### Running a swap experiment

```bash
hs-swap-experiment \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input examples/swap_cases.jsonl \
  --output swap_outputs.jsonl \
  --layers 10 11 12 13 14 15 16 17 18 19 \
  --max-new-tokens 256 \
  --pair-batch-size 4 \
  --dtype bf16 \
  --resume
```

For a CPU smoke test, use a small causal model and an existing layer:

```bash
hs-swap-experiment \
  --model sshleifer/tiny-gpt2 \
  --input examples/swap_cases.jsonl \
  --output tiny_swap_outputs.jsonl \
  --layers 0 \
  --max-new-tokens 8 \
  --device cpu \
  --dtype fp32
```

Sampling is opt-in with `--do-sample --temperature 0.7 --top-p 0.9`.
Remote model code is disabled unless `--trust-remote-code` is explicitly passed.

### Swap output

One result is written per case:

```json
{
  "custom_id": "capital-france",
  "status": "ok",
  "layers": [10, 11],
  "state_position": 42,
  "instruction_token_lengths": [6, 10],
  "filler_counts": [4, 0],
  "baselines": [
    {"instruction_index": 0, "next_token_id": 123, "completion": "..."}
  ],
  "swaps": [
    {
      "source_instruction_index": 1,
      "target_instruction_index": 0,
      "next_token_id": 456,
      "completion": "...",
      "cosine_distance": 0.01,
      "kl_divergence": 0.02
    }
  ],
  "error": null
}
```

With a flat list of N instructions, `baselines` has N entries and `swaps` has
`N × (N - 1)` entries. Grouped inputs contain only cross-group pairs; explicit
pairs contain exactly the requested interventions. Pair generation is chunked
by `--pair-batch-size`. Without `--fail-fast`, a failing case is recorded with
`status: "error"` and processing continues.

## Ordinary batched inference

The inference input is OpenAI-style JSONL:

```json
{"custom_id":"request-1","body":{"messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Say hello."}]}}
```

Plain `prompt` fields and flattened requests are also supported.

```bash
hs-swap-inference \
  --model allura-forge/Llama-3.3-8B-Instruct \
  --input requests.jsonl \
  --output local_outputs.jsonl \
  --batch-size 32 \
  --max-new-tokens 256 \
  --resume
```

`--stop` and `--stop-strings` are aliases. Stop strings are checked only in
newly generated tokens. Each result contains `custom_id`, `raw_output`,
`completion`, `json_output`, and `parse_error`.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover alignment, marker validation, layer discovery, capture, identity
injection, cross-sample swaps, intervened generation, hook cleanup, CLI parsing,
prompt fallback, stopping, EOS handling, decoding, metrics, and JSONL I/O.

A CUDA integration run with `Qwen/Qwen2.5-7B-Instruct` additionally validates
the notebook-style 2x2 grouped experiment: four baselines and all eight directed
cross-group swaps, with simultaneous injection at decoder layers 10 through 19.

## Current limitations

- Alignment applies to the common question suffix and marker position; it does
  not claim semantic one-to-one alignment between different instruction tokens.
- Captured states are decoder-block outputs at `[STATE]`, matching the reference
  notebook. Other intervention sites require an explicit extension.
- Adding a new marker resizes the embedding table. Its initialization is seeded,
  and every baseline and intervention in a run uses the same marker embedding.
- Swap cases are processed one at a time; instruction variants are batched and
  ordered pairs are processed in bounded chunks.
- Hidden vectors are not yet persisted as separate artifacts.
- New model architectures may need an additional decoder-layer resolver path.
