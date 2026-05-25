# Hidden-State Swap for Instruction Causal Analysis

This repository implements a **token-aligned hidden-state swap method** for analyzing the **causal effects of instructions on large language models**.

The core goal is to isolate how *instructional differences* influence internal representations and final generations, while keeping the question itself fixed.

---

## Key Idea

Given the **same question** with **different instructions**:

1. **Baseline Forward Pass**
   - Run each instruction independently
   - Record hidden states at specific token positions and Transformer layers

2. **Hidden-State Swap**
   - Under strict token alignment
   - Inject hidden states from Instruction A into Instruction B’s forward pass (and vice versa)
   - Keep all other computations unchanged

3. **Output Comparison**
   - Observe how internal representation changes propagate to generation behavior
   - Identify where instruction information is encoded across layers

---

## Method Highlights

- **Strict Token Alignment**
  - All instructions are padded to the same token length
  - Ensures semantic alignment of swapped hidden states

- **Layer-wise Controlled Intervention**
  - Arbitrary Transformer layers can be selected for swapping
  - Enables fine-grained causal analysis

- **Minimal Causal Perturbation**
  - Direct hidden-state replacement only
  - No retraining or parameter modification

- **Reproducible and Extensible**
  - Full hidden states are saved
  - Supports downstream probing, similarity analysis, and causal tracing

---
**We are still in the process of debugging this project.**
**We are still in the process of debugging this project.**
**We are still in the process of debugging this project.**
---

## 1) What you can run today

✅ Implemented and runnable:

- Local **batched inference** for *API-style JSONL requests* (OpenAI-like request objects) using HuggingFace Transformers.
- **Streaming output** to JSONL (writes results per batch; no need to wait until completion).
- **Resume mode**: if the output JSONL exists, skip `custom_id`s already written.
- Robust stopping:
  - Supports multiple `eos_token_id`s when available.
  - Optional stop-string criterion that stops **only when all samples in a batch have stopped** (prevents truncation of other samples).

## 2) Project layout

```text
hidden_state_swap_project/
  README.md
  requirements.txt
  pyproject.toml

  src/hs_swap/
    __init__.py
    io.py         # JSONL read/append + resume helper
    models.py     # HF tokenizer/model load, padding-side fixes
    prompting.py  # request -> prompt (chat template preferred)
    inference.py  # batched generate + stop strings + json parse

  scripts/
    run_inference.py   # main runnable CLI
```

## 3) Environment setup

### Requirements

- Python >= 3.9
- CUDA GPU is strongly recommended (CPU will be extremely slow).

### Install dependencies

```bash
cd hidden_state_swap_project
pip install -r requirements.txt
```

(Optional, editable install for development)

```bash
pip install -e .
```

## 4) Input format

`run_inference.py` expects a JSONL file where each line is **one request object**.

### Recommended OpenAI-style request shape

```json
{
  "custom_id": "Mercury_7175875-1",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "..."}
    ],
    "temperature": 0.3,
    "max_tokens": 256
  }
}
```

Also supported:

- Flattened: `{ "custom_id": "...", "messages": [...] }`
- Plain prompt: `{ "custom_id": "...", "body": { "prompt": "..." } }`

Prompt construction rule:

1) If `messages` exist and the tokenizer has `apply_chat_template`, we use it with `add_generation_prompt=True`.
2) Otherwise we fall back to concatenating role/content lines.

## 5) Running batched inference

### Basic usage (LLaMA / decoder-only instruct)

```bash
python scripts/run_inference.py   --model allura-forge/Llama-3.3-8B-Instruct   --input odd_id.jsonl   --output local_outputs.jsonl   --batch-size 32   --max-new-tokens 256   --do-sample   --temperature 0.3   --resume
```

### GLM-4 example

```bash
python scripts/run_inference.py   --model zai-org/GLM-4-9B-0414   --input requests.jsonl   --output glm4_outputs.jsonl   --batch-size 16   --max-new-tokens 256   --do-sample   --temperature 0.3   --resume
```

### Add stop strings (optional)

Stop strings are applied **after decoding**. To avoid truncating other samples, the stop criterion only triggers when **all** samples in the batch have encountered one of the stop strings.

```bash
python scripts/run_inference.py   --model allura-forge/Llama-3.3-8B-Instruct   --input requests.jsonl   --output out.jsonl   --stop-strings "<|eot_id|>" "</s>"
```

## 6) Output format

The output file is a JSONL where each line is one result:

```json
{
  "custom_id": "Mercury_7175875-1",
  "raw_output": "<full decoded text including prompt>",
  "completion": "<decoded text after stripping prompt and stop strings>",
  "json_output": { "task": "...", "verdict": {"A": {...}} },
  "parse_error": null
}
```

Notes:

- `raw_output` is the **entire** decoded string from `model.generate`.
- `completion` removes the prompt prefix and then trims any `--stop-strings`.
- `json_output` is a best-effort parse of either:
  - the entire completion as JSON, or
  - the first `{ ... }` block found inside the completion.

