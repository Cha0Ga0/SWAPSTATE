# StateSwap：探测多项选择题中的支持式—排除式隐藏状态

<p align="right"><a href="README.md">English</a> | 简体中文</p>

> EMNLP 2026 论文 **《StateSwap: Probing Support–Elimination Hidden States in
> Multiple-Choice Questions》** 的代码实现。

## 摘要

大型语言模型面对同一道多项选择题时，如果问题分别采用支持导向和排除导向的措辞，往往会给出不一致的答案。我们研究这种差异是否源于两种措辞所诱导的不同内部表征。为此，我们提出一种双措辞协议：在保持评估目标不变的同时，仅对 prompt 做最小改动，使其分别采用支持导向或排除导向的表述。为了探测内部计算过程，我们追加一个未经训练的特殊 token `[STATE]`，并将其在 residual stream 中的 activation 作为干预接口。在所评估的两个模型中，两种措辞会诱导出可分离的 `[STATE]` activation，且这种差异主要集中在中间层。将这些 activation 在配对 prompt 之间交换，会系统性地改变预测并提高跨措辞的一致性；这提供了基于干预的证据，表明这些 activation 与模型行为具有因果关联。除逐样本替换外，由双措辞对比导出的均值差 steering 方向，相较于匹配的 contrastive activation addition 方向，在本实验协议下呈现出更有界的逐层响应。

## 实现状态

SWAPSTATE 为 HuggingFace decoder-only 语言模型提供可复现的本地推理和 token 对齐隐藏状态干预。

基线推理、标记对齐、层输出捕获、成对隐藏状态注入、干预后生成和 logits 指标均已实现并通过测试。完整的第 10–19 层干预流程也已使用 Qwen2.5-7B-Instruct 在单张 40 GB NVIDIA A100 上验证。

## 功能

- 对 OpenAI 风格 JSONL 请求进行批量本地推理。
- 流式写入输出，并通过 `custom_id` 断点续跑。
- 不受 prompt 干扰的停止字符串，以及严格按 token 边界解码 completion。
- 可配置的单 token `[STATE]` 干预标记。
- 在不同指令变体之间对齐公共问题后缀的 token 位置。
- 在任意从零开始编号的 decoder 层捕获并同时注入隐藏状态。
- 对 N 个变体执行所有有向的 `源指令 → 目标指令` 交换。
- 输出基线/交换后生成、下一 token ID、余弦距离和 KL 散度。
- 支持发现 LLaMA/Qwen/Mistral/Gemma 风格、GPT-2、GPT-NeoX 以及常见 encoder-decoder 路径中的 decoder 层。

## 安装

需要 Python 3.9 或更高版本。7B 及以上模型强烈建议使用 CUDA GPU；小模型测试也支持 CPU。

```bash
conda create --name stateswap python=3.10
conda activate stateswap
pip install -r requirements.txt
pip install -e .
```

安装后提供两个命令：

- `hs-swap-inference`：普通批量生成；
- `hs-swap-experiment`：隐藏状态干预实验。

兼容性入口仍可通过 `python scripts/run_inference.py` 和 `python scripts/run_swap.py` 使用。

## 隐藏状态交换协议

对于每个实验案例：

1. 加载模型前，将 `[STATE]` 注册为一个额外的特殊 token；需要时扩展模型 embedding 表。
2. 对每条指令进行分词，不加入模型自动添加的特殊 token。
3. 在每条指令与公共问题之间插入参与 attention 的单 token 填充符，使每个变体中完整的问题后缀及 `[STATE]` 都位于相同的绝对位置。
4. 通过一次基线 forward，在标记位置捕获各指定 decoder block 的输出。
5. 对每个有向指令对，在全部指定层中用源向量替换目标向量。
6. 注入只在完整 prompt 的 forward/prefill 阶段执行一次。即使发生异常，hook 也始终通过 `finally` 清除。
7. 将干预后的下一 token 分布与目标基线比较，并在同一干预条件下重新生成。

层编号从零开始。一次传入多个层会在同一次 forward 中同时交换这些层，与参考 notebook 的实验方式一致。

### 交换实验输入

JSONL 每行包含一个公共问题和至少两条指令：

```json
{"custom_id":"capital-france","instructions":["Answer with the correct option.","Reason briefly, then answer with the correct option."],"question":"What is the capital of France?","options":{"A":"Paris","B":"Rome","C":"Berlin"}}
```

`options` 可以是对象或数组。也可以通过 `question_block` 提供已经格式化的完整问题。若输入中没有 `[STATE]`，CLI 会自动追加；若出现多个标记则拒绝该输入。参见 `examples/swap_cases.jsonl`。

如需运行 notebook 中的 2×2 设计，请使用 `instruction_groups`：

```json
{"custom_id":"case-2x2","instruction_groups":[["wrong-1","wrong-2"],["correct-1","correct-2"]],"question":"..."}
```

这会生成两个分组之间的双向交换，但排除组内交换。参见 `examples/swap_group_case.jsonl`。显式提供由 `[source_index, target_index]` 构成的 `swap_pairs` 数组，会覆盖上述默认配对方式。

### 运行交换实验

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

如需在 CPU 上进行 smoke test，请使用小型因果语言模型及其实际存在的层：

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

采样默认关闭。可通过 `--do-sample --temperature 0.7 --top-p 0.9` 开启。除非显式传入 `--trust-remote-code`，否则不会执行远程模型代码。

### 交换实验输出

每个案例写入一条结果：

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

对于包含 N 条指令的平铺列表，`baselines` 有 N 项，`swaps` 有 `N × (N - 1)` 项。分组输入只包含跨组配对；显式配对只包含指定干预。配对生成通过 `--pair-batch-size` 分块处理。未传入 `--fail-fast` 时，失败案例会以 `status: "error"` 写入，程序随后继续处理。

## 普通批量推理

推理输入采用 OpenAI 风格 JSONL：

```json
{"custom_id":"request-1","body":{"messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Say hello."}]}}
```

同时支持普通 `prompt` 字段和扁平化请求。

```bash
hs-swap-inference \
  --model allura-forge/Llama-3.3-8B-Instruct \
  --input requests.jsonl \
  --output local_outputs.jsonl \
  --batch-size 32 \
  --max-new-tokens 256 \
  --resume
```

`--stop` 与 `--stop-strings` 是别名。停止字符串只在新生成的 token 中检查。每条结果包含 `custom_id`、`raw_output`、`completion`、`json_output` 和 `parse_error`。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖对齐、标记校验、层发现、捕获、恒等注入、跨样本交换、干预后生成、hook 清理、CLI 解析、prompt fallback、停止条件、EOS 处理、解码、指标和 JSONL I/O。

此外，使用 `Qwen/Qwen2.5-7B-Instruct` 的 CUDA 集成测试验证了 notebook 风格的 2×2 分组实验：4 个基线和全部 8 个有向跨组交换，并在 decoder 第 10–19 层同时注入。

## 当前限制

- 对齐针对公共问题后缀和标记位置，并不表示不同指令 token 之间存在语义上的一一对应。
- 捕获的是 `[STATE]` 位置的 decoder block 输出，与参考 notebook 一致。其他干预位置需要显式扩展。
- 添加新标记会扩展 embedding 表。其初始化受随机种子控制，同一次运行中的所有基线与干预使用相同的标记 embedding。
- 交换案例逐个处理；指令变体采用 batch，所有有向配对按限制大小分块处理。
- 尚未将隐藏向量单独持久化为实验产物。
- 新的模型架构可能需要补充 decoder 层解析路径。
