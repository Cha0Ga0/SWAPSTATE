# Hidden-State Swap for Instruction Causal Analysis

This repository implements a **token-aligned hidden-state swap method** for analyzing the **causal effects of instructions (prompts) on large language models**.

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

## Use Cases

- Instruction tuning and alignment analysis  
- Causal localization of misleading or corrective prompts  
- Prompt strategy comparison at the representation level  
- Layer-wise instruction sensitivity studies  

---

## Project Structure

- `ffqa_hidden_exchange.ipynb`  
  Main experiment notebook, including:
  - Instruction token alignment
  - Hidden-state collection
  - Layer-wise swapping
  - Generation comparison

- Outputs:
  - `*.jsonl`: baseline and swapped generations
  - `*.npy`: cached hidden states for further analysis

---

## Requirements

- Python ≥ 3.9  
- PyTorch  
- HuggingFace Transformers  
- GPU recommended

It implements:

- **Strict instruction token alignment** via `instr + "\n" + [SPACE]*pad + "\n"` so that the question span starts at the same token index.
- A **single `[STATE]` token** appended to each question (`Question + "\n[STATE]"`), used as the intervention anchor.
- **Hidden-state capture** at the `[STATE]` token position for selected Transformer layers.
- **Hidden-state swapping** across two instructions (2-sample batch), injecting the swapped vectors at the same token position and layers.
- Output logging to `*.jsonl` and cached state vectors to `*.npy`.

---

## Install

```bash
pip install -r requirements.txt
# or editable
pip install -e .
```

> GPU is strongly recommended.

---

## Prepare data

Use a CSV file with a `Question` column (and optional TruthfulQA metadata columns like `Type`, `Category`, etc.).

Example: `TruthfulQA.csv`

---

## Run (reproduces the notebook main loop)

```bash
python scripts/run_truthfulqa_pair_swap.py   --model Qwen/Qwen2.5-7B-Instruct   --input-csv TruthfulQA.csv   --output-jsonl swapseed123_truthfulqa_qwen25_pair.jsonl   --state-npy stateseed123_truthfulqa_qwen25_pair.npy   --instruction-1 "Instruction: First, privately develop the accurate reasoning process, but do not disclose any reasoning. Then provide the shortest, factually correct final answer in no more than two sentences."   --instruction-2 "Instruction: First, internally remove common erroneous reasoning paths, but do not reveal any reasoning. Then output the most concise, factually correct final answer in at most two sentences."   --layers-to-swap 10-19   --pad-token-limit 64   --max-new-tokens 128
```

### Optional knobs (same semantics as the notebook)

- `--do-sample` enables sampling generation (default: greedy, `DO_SAMPLE=False`).
- `--top-p` nucleus sampling parameter (default: `0.0`).
- `--limit-n 100` to run only the first N rows (like `LIMIT_N`).
- `--hf-endpoint https://hf-mirror.com` and `--hf-home /root/autodl-tmp/cache` to mimic the notebook's HF mirror/cache.
- `--seed 123` sets Python/NumPy/Torch seeds (like the notebook).

---

## Outputs

### JSONL output (`--output-jsonl`)

Each line is a record for one dataset row, containing:

- Original metadata (`row_id`, `Question`, etc.)
- Baseline generations for the two instructions (`baseline1_response`, `baseline2_response`)
- Swapped generations (`swap_2_to_1_response`, `swap_1_to_2_response`)
- Logits-distance metrics (`swap_*_cos`, `swap_*_kl`)

### NPY cache (`--state-npy`)

`(num_rows*2, num_layers, hidden_dim)` array of baseline `[STATE]` hidden vectors, saved in the same order as the notebook:

- For each row: first instr1 baseline, then instr2 baseline.

---

## Code map to the notebook

- Notebook tokenizer/model setup → `src/ffqa_swap/modeling.py`
- Instruction alignment (`tokenize_instructions_and_align`) → `src/ffqa_swap/alignment.py`
- Baseline forward + hook capture → `src/ffqa_swap/forward.py::forward_with_capture_batch_single_pos`
- Injected forward + swap + metrics → `src/ffqa_swap/forward.py::forward_with_injected_state_batch_single_pos`
- CSV reading → `src/ffqa_swap/data.py`
- Main loop (per-row build, check state_pos, baseline, swap, write JSONL, save NPY) → `src/ffqa_swap/runner.py`
- CLI wrapper → `scripts/run_truthfulqa_pair_swap.py`

---

## Notes / Assumptions (same as notebook)

- `[STATE]` must tokenize to **exactly one token**.
- `" "` must tokenize to **exactly one token** (needed for token-aligned space padding).
- The code assumes Qwen-like architecture where decoder layers are accessible as `model.model.layers[...]`.


