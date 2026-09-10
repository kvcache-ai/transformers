# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import unittest
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from transformers import PreTrainedConfig, PreTrainedModel
from transformers.integrations.kimi_compat import prepare_remote_kimi_model_class
from transformers.modeling_layers import GradientCheckpointingLayer


class KimiCompatibilityTest(unittest.TestCase):
    def setUp(self):
        module = ModuleType("legacy_kimi_test_model")

        class Config(PreTrainedConfig):
            model_type = "kimi_k25"

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, hidden_states):
                return self.linear(hidden_states).sin()

        class TextModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gradient_checkpointing = False
                self.layer = module.DeepseekV3DecoderLayer()

            def forward(self, hidden_states, use_cache=True, past_key_values=None):
                return self.layer(hidden_states), use_cache, past_key_values

        class Encoder:
            def __init__(self):
                self.observed = self.use_deterministic_attn

        class Model(PreTrainedModel):
            config_class = Config
            main_input_name = "inputs_embeds"

            def __init__(self):
                super().__init__(Config())
                self.model = module.DeepseekV3Model()

        for cls in (Decoder, TextModel, Model):
            cls.__module__ = module.__name__
        module.DeepseekV3DecoderLayer = Decoder
        module.DeepseekV3Model = TextModel
        module.DeepseekV3ForCausalLM = Model
        module.MoonViT3dEncoder = Encoder
        self.module = module
        self.model_class = Model
        self.module_patch = patch.dict(sys.modules, {module.__name__: module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_standard_checkpoint_context_gradient_parity_and_disable(self):
        prepare_remote_kimi_model_class(self.model_class)
        model = self.model_class().train()
        inputs = torch.randn(2, 4, requires_grad=True)
        keys = set(model.state_dict())
        expected = model.model(inputs, False)[0]
        expected.sum().backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        contexts = []

        @contextmanager
        def record(phase):
            contexts.append(phase)
            yield

        model.gradient_checkpointing_enable(
            {"use_reentrant": False, "context_fn": lambda: (record("forward"), record("recompute"))}
        )
        output, use_cache, past = model.model(inputs, True, object())
        self.assertFalse(use_cache)
        self.assertIsNone(past)
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        for param, expected_grad in zip(model.parameters(), gradients):
            torch.testing.assert_close(param.grad, expected_grad)
        self.assertEqual(contexts, ["forward", "recompute"])
        self.assertEqual(set(model.state_dict()), keys)
        model.eval()
        self.assertTrue(model.model(inputs)[1])
        model.train()
        model.gradient_checkpointing_disable()
        self.assertFalse(model.model.layer.gradient_checkpointing)
        self.assertTrue(model.model(inputs)[1])

    def test_idempotent_and_preserves_decoder_name(self):
        original_name = self.module.DeepseekV3DecoderLayer.__name__
        self.assertIs(prepare_remote_kimi_model_class(self.model_class), self.model_class)
        decoder = self.module.DeepseekV3DecoderLayer
        forward = self.module.DeepseekV3Model.forward
        prepare_remote_kimi_model_class(self.model_class)
        self.assertIs(decoder, self.module.DeepseekV3DecoderLayer)
        self.assertIs(forward, self.module.DeepseekV3Model.forward)
        self.assertEqual(decoder.__name__, original_name)
        self.assertTrue(issubclass(decoder, GradientCheckpointingLayer))
        self.assertFalse(self.module.MoonViT3dEncoder().observed)

    def test_non_kimi_class_unchanged(self):
        other = SimpleNamespace(config_class=SimpleNamespace(model_type="qwen3_moe"))
        self.assertIs(prepare_remote_kimi_model_class(other), other)
        self.assertFalse(hasattr(other, "supports_gradient_checkpointing"))

    def test_existing_vision_default_unchanged(self):
        self.module.MoonViT3dEncoder.use_deterministic_attn = True
        prepare_remote_kimi_model_class(self.model_class)
        self.assertTrue(self.module.MoonViT3dEncoder().observed)
