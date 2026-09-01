import unittest

from hs_swap.swap_cli import _extract_case, parse_args


class SwapCliTests(unittest.TestCase):
    def test_flat_instructions_create_all_ordered_pairs(self):
        case = _extract_case(
            {
                "custom_id": "flat",
                "instructions": ["one", "two", "three"],
                "question": "Question",
            },
            "[STATE]",
        )
        self.assertEqual(len(case.swap_pairs), 6)
        self.assertIn((0, 1), case.swap_pairs)
        self.assertIn((1, 0), case.swap_pairs)
        self.assertTrue(case.question_block.endswith("[STATE]"))

    def test_instruction_groups_only_create_cross_group_pairs(self):
        case = _extract_case(
            {
                "custom_id": "groups",
                "instruction_groups": [["a", "b"], ["c", "d"]],
                "question_block": "Question\n[STATE]",
            },
            "[STATE]",
        )
        self.assertEqual(case.instructions, ["a", "b", "c", "d"])
        self.assertEqual(len(case.swap_pairs), 8)
        self.assertNotIn((0, 1), case.swap_pairs)
        self.assertIn((0, 2), case.swap_pairs)
        self.assertIn((2, 0), case.swap_pairs)

    def test_explicit_pairs_override_defaults(self):
        case = _extract_case(
            {
                "custom_id": "explicit",
                "instructions": ["a", "b", "c"],
                "swap_pairs": [[2, 0]],
                "question": "Question",
            },
            "[STATE]",
        )
        self.assertEqual(case.swap_pairs, [(2, 0)])

    def test_pair_batch_size_must_be_positive(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--model",
                    "model",
                    "--input",
                    "input.jsonl",
                    "--output",
                    "output.jsonl",
                    "--layers",
                    "0",
                    "--pair-batch-size",
                    "0",
                ]
            )


if __name__ == "__main__":
    unittest.main()
