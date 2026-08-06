# Copyright 2026-present the HuggingFace Inc. team.
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

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from transformers.integrations.kt import (
    HfTrainerKTConfig,
    _validate_kt_int8_loading_info,
    _validate_kt_prequantized_loading_info,
    is_kt_fp8_expert_loading_enabled,
    is_kt_prequantized_expert_loading_enabled,
    unset_kt_config,
)
from transformers.modeling_utils import get_total_byte_count
from transformers.utils.loading_report import LoadStateDictInfo


def _loading_info(**overrides):
    values = {
        "missing_keys": set(),
        "unexpected_keys": set(),
        "mismatched_keys": set(),
        "error_msgs": [],
        "conversion_errors": {},
    }
    values.update(overrides)
    return LoadStateDictInfo(**values)


class KTInt8LoadingValidationTest(unittest.TestCase):
    def tearDown(self):
        unset_kt_config()

    def enable_int8(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "int8",
            }
        )

    def test_allows_filtered_experts_and_deepseek_mtp_tail(self):
        self.enable_int8()
        info = _loading_info(
            missing_keys={
                "model.layers.3.mlp.experts.0.gate_proj.weight",
                "model.layers.3.mlp.experts.gate_up_proj",
            },
            unexpected_keys={
                "model.layers.61.self_attn.q_proj.weight",
                "model.layers.61.mlp.gate.weight",
            },
        )

        deepseek_model = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v3", num_hidden_layers=61))
        _validate_kt_int8_loading_info(info, deepseek_model)

    def test_rejects_mtp_shaped_unexpected_key_for_other_models(self):
        self.enable_int8()
        other_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe", num_hidden_layers=61))

        with self.assertRaisesRegex(RuntimeError, "unexpected_keys"):
            _validate_kt_int8_loading_info(
                _loading_info(unexpected_keys={"model.layers.61.self_attn.q_proj.weight"}),
                other_model,
            )

    def test_rejects_each_non_expert_loading_failure(self):
        self.enable_int8()
        failures = {
            "missing_keys": {"model.layers.3.self_attn.q_proj.weight"},
            "unexpected_keys": {
                "model.layers.3.mlp.experts.unrecognized.weight",
                "model.layers.60.unrecognized.weight",
            },
            "mismatched_keys": {
                ("model.embed_tokens.weight", (8, 16), (8, 32)),
            },
            "conversion_errors": {"model.norm.weight": "dequantization failed"},
            "error_msgs": ["invalid checkpoint tensor"],
        }

        for field, value in failures.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, field):
                    _validate_kt_int8_loading_info(_loading_info(**{field: value}))

    def test_non_int8_loading_is_unchanged(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "bf16",
            }
        )

        _validate_kt_int8_loading_info(
            _loading_info(
                missing_keys={"model.layers.3.self_attn.q_proj.weight"},
                unexpected_keys={"model.layers.60.unrecognized.weight"},
            )
        )

    def test_int8_environment_configuration_is_strict(self):
        with patch.dict(
            "os.environ",
            {
                "ACCELERATE_USE_KT": "true",
                "ACCELERATE_KT_SKIP_EXPERT_LOADING": "true",
                "ACCELERATE_KT_EXPERT_WEIGHT_FORMAT": "int8",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "missing_keys"):
                _validate_kt_int8_loading_info(_loading_info(missing_keys={"model.layers.3.self_attn.q_proj.weight"}))

    def test_fp8_uses_the_same_strict_non_expert_contract(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "fp8",
            }
        )

        self.assertTrue(is_kt_fp8_expert_loading_enabled())
        self.assertTrue(is_kt_prequantized_expert_loading_enabled())
        _validate_kt_prequantized_loading_info(
            _loading_info(missing_keys={"model.layers.3.mlp.experts.0.gate_proj.weight"})
        )
        with self.assertRaisesRegex(RuntimeError, "KT FP8.*missing_keys"):
            _validate_kt_prequantized_loading_info(
                _loading_info(missing_keys={"model.layers.3.self_attn.q_proj.weight"})
            )

    def test_fp8_lora_dropout_is_loaded_from_environment(self):
        with patch.dict(
            "os.environ",
            {
                "ACCELERATE_USE_KT": "true",
                "ACCELERATE_KT_EXPERT_WEIGHT_FORMAT": "fp8",
                "ACCELERATE_KT_LORA_DROPOUT": "0.125",
            },
            clear=False,
        ):
            self.kt_config = HfTrainerKTConfig({"enabled": True})

        self.assertAlmostEqual(self.kt_config.kt_lora_dropout, 0.125)

    def test_fp8_routed_experts_are_excluded_from_allocator_warmup(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "fp8",
            }
        )
        parameters = {
            "model.layers.3.mlp.experts.0.gate_proj.weight": torch.empty(1024, dtype=torch.bfloat16),
            "model.layers.3.self_attn.q_proj.weight": torch.empty(8, dtype=torch.bfloat16),
        }
        device_map = dict.fromkeys(parameters, "cuda:0")

        for architecture in (
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "Qwen2MoeForCausalLM",
            "Qwen3MoeForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
            "Glm4MoeForCausalLM",
            "MixtralForCausalLM",
        ):
            with self.subTest(architecture=architecture):
                model = SimpleNamespace(
                    all_tied_weights_keys={},
                    config=SimpleNamespace(architectures=[architecture]),
                    _tp_plan=None,
                    get_parameter_or_buffer=parameters.__getitem__,
                )

                warmup_bytes = get_total_byte_count(model, device_map)

                self.assertEqual(warmup_bytes["cuda:0"], 16)


if __name__ == "__main__":
    unittest.main()
