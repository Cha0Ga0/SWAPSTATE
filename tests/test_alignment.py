import unittest

import torch

from hs_swap.alignment import build_aligned_inputs, ensure_state_marker


class CharacterTokenizer:
    bos_token_id = 7

    def __init__(self):
        self.vocab = {" ": 1, "\n": 2, "[STATE]": 3}
        self.next_id = 10

    def __call__(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("tests expect add_special_tokens=False")
        ids = []
        index = 0
        while index < len(text):
            if text.startswith("[STATE]", index):
                ids.append(self.vocab["[STATE]"])
                index += len("[STATE]")
                continue
            char = text[index]
            if char not in self.vocab:
                self.vocab[char] = self.next_id
                self.next_id += 1
            ids.append(self.vocab[char])
            index += 1
        return {"input_ids": ids}


class AlignmentTests(unittest.TestCase):
    def test_suffix_and_state_marker_are_token_aligned(self):
        tokenizer = CharacterTokenizer()
        aligned = build_aligned_inputs(
            tokenizer,
            ["A", "LONG"],
            "Q[STATE]",
        )

        self.assertEqual(aligned.filler_counts, [3, 0])
        self.assertEqual(aligned.input_ids.shape[0], 2)
        self.assertTrue(torch.all(aligned.attention_mask == 1))
        self.assertTrue(
            torch.all(
                aligned.input_ids[:, aligned.state_position]
                == aligned.state_token_id
            )
        )
        question_start = aligned.state_position - 1
        self.assertTrue(
            torch.equal(
                aligned.input_ids[0, question_start:],
                aligned.input_ids[1, question_start:],
            )
        )

    def test_marker_is_appended_once(self):
        self.assertEqual(
            ensure_state_marker("Question", "[STATE]"),
            "Question\n[STATE]",
        )
        self.assertEqual(
            ensure_state_marker("Question\n[STATE]", "[STATE]"),
            "Question\n[STATE]",
        )
        with self.assertRaises(ValueError):
            ensure_state_marker("[STATE] x [STATE]", "[STATE]")


if __name__ == "__main__":
    unittest.main()
