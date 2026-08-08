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
from unittest.mock import Mock, patch

import torch

from transformers import BertConfig, BertModel
from transformers.integrations.kt import (
    HfTrainerKTConfig,
    _validate_kt_int8_loading_info,
    _validate_kt_prequantized_loading_info,
    is_kt_fp8_expert_loading_enabled,
    is_kt_prequantized_expert_loading_enabled,
    unset_kt_config,
)
from transformers.modeling_utils import PreTrainedModel, get_total_byte_count
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

    def test_wrapper_does_not_import_kernel_fields_from_environment(self):
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

        with self.assertRaises(AttributeError):
            _ = self.kt_config.kt_lora_dropout

    def test_bf16_claim_precedes_device_map_without_a_load_plan(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "bf16",
            }
        )
        config = BertConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            vocab_size=32,
        )
        source = BertModel(config)
        events = []

        def claim(model):
            events.append(("claim", model))
            unset_kt_config()
            return ()

        def get_device_map(model, device_map, max_memory, hf_quantizer):
            events.append(("device_map", model))
            return device_map

        with (
            patch("transformers.integrations.kt_artifacts.claim_kt_routed_expert_subtrees", side_effect=claim),
            patch("transformers.modeling_utils._get_device_map", side_effect=get_device_map),
        ):
            loaded = BertModel.from_pretrained(
                None,
                config=config,
                state_dict=source.state_dict(),
                device_map={"": "cpu"},
            )

        self.assertEqual([event for event, _ in events], ["claim", "device_map"])
        self.assertIs(events[0][1], loaded)
        self.assertIs(events[1][1], loaded)

    def test_kt_finalize_keeps_ordinary_missing_meta_in_the_standard_path(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "bf16",
            }
        )
        expert_key = "model.layers.0.mlp.experts.gate_up_proj"
        ordinary_key = "model.layers.0.self_attn.q_proj.weight"
        expert_module = SimpleNamespace(
            gate_up_proj=torch.nn.Parameter(torch.empty(2, 2, device="meta")),
        )
        model = SimpleNamespace(
            get_submodule=Mock(return_value=expert_module),
            mark_tied_weights_as_initialized=Mock(),
            _move_missing_keys_from_meta_to_device=Mock(),
            _initialize_missing_keys=Mock(),
            tie_weights=Mock(),
            _adjust_missing_and_unexpected_keys=Mock(),
        )
        load_config = SimpleNamespace(
            device_map=None,
            device_mesh=None,
            hf_quantizer=None,
            is_quantized=False,
            pretrained_model_name_or_path="test-model",
            ignore_mismatched_sizes=False,
        )
        loading_info = _loading_info(missing_keys={expert_key, ordinary_key})

        with (
            patch(
                "transformers.integrations.kt.is_kt_routed_expert_parameter_name",
                side_effect=lambda name: name == expert_key,
            ),
            patch("transformers.modeling_utils.log_state_dict_report"),
        ):
            result = PreTrainedModel._finalize_model_loading(model, load_config, loading_info)

        self.assertIs(result, loading_info)
        self.assertEqual(loading_info.missing_keys, {ordinary_key})
        self.assertEqual(expert_module.gate_up_proj.device.type, "cpu")
        model._move_missing_keys_from_meta_to_device.assert_called_once_with(
            {ordinary_key},
            None,
            None,
            None,
        )

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

    def test_fsdp_rank_zero_fill_delegates_routed_expert_names_to_kt(self):
        expert_name = "model.layers.3.mlp.experts.new_layout.weight"
        normal_name = "model.norm.weight"
        model = SimpleNamespace(
            named_parameters=lambda: [
                (expert_name, torch.empty(2, 2, device="meta")),
                (normal_name, torch.empty(2, device="meta")),
            ],
            named_buffers=lambda: [],
        )

        with (
            patch("transformers.modeling_utils.is_deepspeed_zero3_enabled", return_value=False),
            patch("transformers.modeling_utils.is_fsdp_enabled", return_value=True),
            patch("transformers.modeling_utils.is_local_dist_rank_0", return_value=False),
            patch("transformers.integrations.kt.is_kt_expert_loading_enabled", return_value=True),
            patch(
                "transformers.integrations.kt.is_kt_routed_expert_parameter_name",
                side_effect=lambda name: name == expert_name,
            ) as is_routed,
            patch("transformers.modeling_utils._load_parameter_into_model") as load_parameter,
        ):
            PreTrainedModel._move_missing_keys_from_meta_to_device(model, set(), None, None, None)

        self.assertEqual([call.args[0] for call in is_routed.call_args_list], [expert_name, normal_name])
        load_parameter.assert_called_once()
        self.assertEqual(load_parameter.call_args.args[:2], (model, normal_name))


if __name__ == "__main__":
    unittest.main()
