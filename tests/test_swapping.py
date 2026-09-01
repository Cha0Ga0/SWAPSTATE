import unittest
from types import SimpleNamespace

import torch
from torch import nn

from hs_swap.swapping import (
    capture_hidden_states,
    forward_with_injected_states,
    generate_with_injected_states,
    get_input_device,
    resolve_decoder_layers,
    select_state_vectors,
)


class MixingLayer(nn.Module):
    def forward(self, hidden_states):
        mixed = hidden_states + hidden_states.mean(dim=1, keepdim=True)
        return (mixed,)


class TinyHookModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.embedding = nn.Embedding(32, 6)
        decoder = nn.Module()
        decoder.layers = nn.ModuleList([MixingLayer(), MixingLayer()])
        self.model = decoder
        self.lm_head = nn.Linear(6, 11, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=self.lm_head(hidden))

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **kwargs):
        del kwargs
        output_ids = input_ids
        for _ in range(max_new_tokens):
            outputs = self.forward(output_ids, attention_mask=attention_mask, use_cache=True)
            next_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            output_ids = torch.cat((output_ids, next_ids), dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_ids)),
                    dim=1,
                )
        return output_ids


class SwappingTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyHookModel().eval()
        self.inputs = {
            "input_ids": torch.tensor([[1, 3, 4], [2, 3, 4]]),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
        }

    def test_layer_resolution_and_input_device(self):
        self.assertEqual(len(resolve_decoder_layers(self.model)), 2)
        self.assertEqual(get_input_device(self.model), torch.device("cpu"))

    def test_capture_and_identity_injection_match(self):
        baseline = capture_hidden_states(self.model, self.inputs, 1, [0])
        identity = forward_with_injected_states(
            self.model,
            self.inputs,
            1,
            baseline.state_vectors,
            baseline.logits,
        )

        self.assertEqual(baseline.state_vectors[0].shape, (2, 6))
        self.assertTrue(torch.allclose(identity.logits, baseline.logits, atol=1e-6))
        self.assertTrue(torch.all(identity.cosine_distances.abs() < 1e-6))
        self.assertTrue(torch.all(identity.kl_divergences.abs() < 1e-6))

    def test_cross_sample_swap_changes_downstream_logits_and_generation(self):
        baseline = capture_hidden_states(self.model, self.inputs, 1, [0])
        swapped_states = select_state_vectors(baseline.state_vectors, [1, 0])
        swapped = forward_with_injected_states(
            self.model,
            self.inputs,
            1,
            swapped_states,
            baseline.logits,
        )
        generated = generate_with_injected_states(
            self.model,
            self.inputs,
            1,
            swapped_states,
            max_new_tokens=1,
        )

        self.assertFalse(torch.allclose(swapped.logits, baseline.logits))
        self.assertTrue(torch.all(swapped.cosine_distances > 0))
        self.assertTrue(torch.all(swapped.kl_divergences > 0))
        self.assertEqual(generated.shape, (2, 4))

    def test_hooks_are_removed_after_an_error(self):
        baseline = capture_hidden_states(self.model, self.inputs, 1, [0])
        bad_states = {0: torch.zeros((1, 6))}
        with self.assertRaises(ValueError):
            forward_with_injected_states(
                self.model,
                self.inputs,
                1,
                bad_states,
                baseline.logits,
            )
        self.assertEqual(len(self.model.model.layers[0]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
