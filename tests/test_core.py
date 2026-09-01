import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from hs_swap.cli import _has_pending_requests, parse_args
from hs_swap.inference import (
    BatchStopOnStrings,
    _collect_eos_token_ids,
    generate_batch,
    postprocess_stop_strings,
    try_parse_json,
)
from hs_swap.io import append_jsonl, read_jsonl
from hs_swap.models import _pick_dtype
from hs_swap.prompting import build_prompt_from_request


class CliTests(unittest.TestCase):
    def test_stop_strings_alias_and_safe_remote_code_default(self):
        args = parse_args(
            [
                "--model",
                "model",
                "--input",
                "input.jsonl",
                "--output",
                "output.jsonl",
                "--stop-strings",
                "one",
                "two",
            ]
        )
        self.assertEqual(args.stop_strings, ["one", "two"])
        self.assertFalse(args.trust_remote_code)

    def test_resume_preflight_detects_pending_requests(self):
        with tempfile.TemporaryDirectory() as root:
            input_path = os.path.join(root, "input.jsonl")
            append_jsonl(input_path, {"custom_id": "done"})
            self.assertFalse(
                _has_pending_requests(input_path, {"done"})
            )
            append_jsonl(input_path, {"custom_id": "pending"})
            self.assertTrue(
                _has_pending_requests(input_path, {"done"})
            )


class PromptingTests(unittest.TestCase):
    def test_messages_use_chat_template_when_configured(self):
        class Tokenizer:
            chat_template = "configured"

            def apply_chat_template(self, messages, **kwargs):
                self.call = (messages, kwargs)
                return "templated"

        tokenizer = Tokenizer()
        request = {"body": {"messages": [{"role": "user", "content": "Hi"}]}}

        self.assertEqual(build_prompt_from_request(tokenizer, request), "templated")
        self.assertTrue(tokenizer.call[1]["add_generation_prompt"])

    def test_messages_fall_back_when_chat_template_is_missing(self):
        class Tokenizer:
            chat_template = None

            def apply_chat_template(self, messages, **kwargs):
                raise AssertionError("apply_chat_template must not be called")

        request = {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ]
        }

        self.assertEqual(
            build_prompt_from_request(Tokenizer(), request),
            "[system] Be concise.\n[user] Hi\n[assistant]",
        )


class DecodeTokenizer:
    pieces = {0: "", 1: "<STOP>", 2: "ok", 10: "P", 11: "Q", 20: "A", 21: "B"}
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, prompts, **kwargs):
        del prompts, kwargs
        return {
            "input_ids": torch.tensor([[10, 11], [0, 11]]),
            "attention_mask": torch.tensor([[1, 1], [0, 1]]),
        }

    def batch_decode(self, rows, **kwargs):
        del kwargs
        return ["".join(self.pieces.get(int(token), "?") for token in row) for row in rows]


class InferenceTests(unittest.TestCase):
    def test_stop_strings_ignore_the_prompt(self):
        criterion = BatchStopOnStrings(
            ["<STOP>"],
            DecodeTokenizer(),
            prompt_length=2,
        )
        scores = torch.empty(0)

        prompt_only_stop = torch.tensor([[1, 1, 2], [1, 1, 2]])
        self.assertFalse(criterion(prompt_only_stop, scores))

        generated_stop = torch.tensor([[1, 1, 1], [1, 1, 1]])
        self.assertTrue(criterion(generated_stop, scores))

    def test_only_configured_eos_ids_are_collected(self):
        tokenizer = SimpleNamespace(
            eos_token_id=3,
            additional_special_tokens_ids=[90, 91],
        )
        model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=[2, 3])
        )
        self.assertEqual(_collect_eos_token_ids(tokenizer, model), [2, 3])

    def test_generate_batch_decodes_completion_by_token_boundary(self):
        class Model:
            device = torch.device("cpu")
            generation_config = SimpleNamespace(eos_token_id=99)

            def generate(self, input_ids, attention_mask, **kwargs):
                del attention_mask, kwargs
                suffix = torch.tensor([[20, 21], [20, 21]])
                return torch.cat((input_ids, suffix), dim=1)

        loaded = SimpleNamespace(tokenizer=DecodeTokenizer(), model=Model())
        results = generate_batch(loaded, ["one", "two"], do_sample=False)

        self.assertEqual([result.completion for result in results], ["AB", "AB"])
        self.assertEqual(results[0].raw_output, "PQAB")
        self.assertEqual(results[1].raw_output, "QAB")

    def test_json_and_stop_postprocessing(self):
        self.assertEqual(
            postprocess_stop_strings("answer<STOP>ignored", ["<STOP>"]),
            "answer",
        )
        parsed, error = try_parse_json('prefix {"ok": true} suffix')
        self.assertEqual(parsed, {"ok": True})
        self.assertIsNone(error)
        parsed, error = try_parse_json(
            'first {"number": 1} second {"number": 2}'
        )
        self.assertEqual(parsed, {"number": 1})
        self.assertIsNone(error)


class ModelAndIoTests(unittest.TestCase):
    def test_auto_dtype_is_fp32_on_cpu(self):
        self.assertEqual(_pick_dtype("auto", "cpu"), torch.float32)

    def test_append_jsonl_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, "nested", "output.jsonl")
            append_jsonl(output, {"custom_id": "one"})
            self.assertEqual(read_jsonl(output), [{"custom_id": "one"}])


if __name__ == "__main__":
    unittest.main()
