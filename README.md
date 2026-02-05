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
