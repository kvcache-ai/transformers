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

import gc
import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from transformers import TrainingArguments
from transformers.integrations.kt import (
    HfTrainerKTConfig,
    _get_kt_config,
    configure_kt,
    is_kt_expert_loading_enabled,
    unset_kt_config,
)
from transformers.trainer import Trainer


def kt_owned_config(config_type):
    config_type.__module__ = "kt_kernel.sft.config"
    return config_type


class TrainingArgumentsKTConfigTest(unittest.TestCase):
    def tearDown(self):
        unset_kt_config()
        os.environ.pop("ACCELERATE_USE_KT", None)

    def make_args(self):
        output_dir = tempfile.mkdtemp()
        with patch.dict(os.environ, {"ACCELERATE_USE_KT": "false"}):
            return TrainingArguments(output_dir=output_dir)

    def test_update_is_atomic_and_does_not_mutate_input(self):
        args = self.make_args()
        source = {"kt_backend": "AMXBF16", "kt_lora_rank": 8}

        returned = args.update_kt_config(source, adapter_name_or_path="/tmp/adapter")

        self.assertIs(returned, args)
        self.assertEqual(source, {"kt_backend": "AMXBF16", "kt_lora_rank": 8})
        self.assertIsNot(args.kt_config, source)
        self.assertEqual(args.kt_config, source)
        self.assertTrue(args.hf_kt_config.enabled)
        self.assertTrue(args.hf_kt_config.kt_skip_expert_loading)
        self.assertIs(args.kt_config, args.hf_kt_config.config)
        self.assertIs(args.kt_config, args.accelerator_config.kt_config)
        self.assertEqual(args.kt_adapter_name_or_path, "/tmp/adapter")

    def test_configure_kt_copies_mapping_and_keeps_state_while_handle_is_alive(self):
        source = {"kt_backend": "AMXBF16", "kt_activation_policy": {"cpu": "retain"}}

        handle = configure_kt(source)
        source["kt_backend"] = "changed"
        source["kt_activation_policy"]["cpu"] = "recompute"

        self.assertEqual(handle.kt_backend, "AMXBF16")
        self.assertEqual(handle.kt_activation_policy, {"cpu": "retain"})
        self.assertIs(_get_kt_config(), handle)
        self.assertEqual(os.environ["ACCELERATE_USE_KT"], "true")

        del handle
        gc.collect()

        self.assertIsNone(_get_kt_config())
        self.assertNotIn("ACCELERATE_USE_KT", os.environ)

    def test_configure_kt_accepts_json_path_and_typed_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as config_file:
            json.dump({"kt_backend": "AMXBF16"}, config_file)
            config_file.flush()
            json_handle = configure_kt(config_file.name)

        self.assertEqual(json_handle.kt_backend, "AMXBF16")

        @kt_owned_config
        @dataclass
        class KTConfig:
            kt_backend: str = "auto"
            kt_lora_rank: int = 8

        typed_config = KTConfig()
        typed_handle = configure_kt(typed_config)

        self.assertEqual(typed_config, KTConfig())
        self.assertIs(typed_handle.config, typed_config)
        self.assertIs(_get_kt_config(), typed_handle)

    def test_typed_config_contract_rejects_name_only_classes_and_class_objects(self):
        @dataclass
        class KTConfig:
            kt_backend: str = "AMXBF16"

        with self.assertRaisesRegex(TypeError, "kt_kernel.sft.config.KTConfig"):
            configure_kt(KTConfig())

        KTConfig.__module__ = "kt_kernel.sft.config"
        with self.assertRaisesRegex(TypeError, "an instance of kt_kernel.sft.config.KTConfig"):
            configure_kt(KTConfig)

        @dataclass
        class Impostor:
            kt_backend: str = "AMXBF16"

        Impostor.__module__ = "kt_kernel.sft.config"
        with self.assertRaisesRegex(TypeError, "got .*Impostor"):
            configure_kt(Impostor())

    def test_typed_config_identity_survives_runtime_metadata_and_activation_environment(self):
        post_init_calls = []

        @kt_owned_config
        @dataclass
        class KTConfig:
            kt_activation_policy: dict[str, str] | None = None
            kt_checkpoint_files: list[str] | None = None
            kt_sharded_metadata: dict | None = None

            def __post_init__(self):
                post_init_calls.append(self)
                if self.kt_activation_policy is not None and os.environ.get("ACCELERATE_KT_ACTIVATION_POLICY"):
                    raise ValueError("activation policy was reconstructed under a conflicting environment")

        config = KTConfig(kt_activation_policy={"cpu": "retain", "gpu": "recompute"})
        with patch.dict(
            os.environ,
            {"ACCELERATE_KT_ACTIVATION_POLICY": '{"cpu":"recompute","gpu":"recompute"}'},
        ):
            handle = configure_kt(config)
            handle.set_runtime_metadata(
                kt_checkpoint_files=["model-00001-of-00002.safetensors"],
                kt_sharded_metadata={"weight_map": {"model.layers.0.mlp.experts": "model-00001-of-00002.safetensors"}},
            )

            nested_config = handle.config
            resolved_config = nested_config if isinstance(nested_config, KTConfig) else KTConfig(**nested_config)

        self.assertIs(nested_config, config)
        self.assertIs(resolved_config, config)
        self.assertEqual(post_init_calls, [config])
        self.assertEqual(handle.kt_checkpoint_files, ["model-00001-of-00002.safetensors"])
        self.assertEqual(
            handle.kt_sharded_metadata,
            {"weight_map": {"model.layers.0.mlp.experts": "model-00001-of-00002.safetensors"}},
        )

    def test_ordinary_arguments_supersede_live_transformers_owned_kt_state(self):
        kt_args = TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            kt_config={"kt_backend": "AMXBF16"},
        )
        kt_handle = kt_args.hf_kt_config
        kt_mapping = kt_args.kt_config
        self.assertEqual(os.environ["ACCELERATE_USE_KT"], "true")

        ordinary_args = TrainingArguments(output_dir=tempfile.mkdtemp())

        self.assertIs(kt_args.hf_kt_config, kt_handle)
        self.assertIs(kt_args.kt_config, kt_mapping)
        self.assertFalse(hasattr(ordinary_args, "hf_kt_config"))
        self.assertIsNone(ordinary_args.kt_config)
        self.assertIsNone(_get_kt_config())
        self.assertFalse(is_kt_expert_loading_enabled())
        self.assertNotIn("ACCELERATE_USE_KT", os.environ)

    def test_external_environment_still_enables_ordinary_arguments(self):
        with patch.dict(os.environ, {"ACCELERATE_USE_KT": "true"}):
            args = TrainingArguments(output_dir=tempfile.mkdtemp())

            self.assertTrue(args.hf_kt_config.enabled)
            self.assertIs(_get_kt_config(), args.hf_kt_config)
            unset_kt_config()
            self.assertEqual(os.environ["ACCELERATE_USE_KT"], "true")

    def test_post_init_keeps_raw_mapping_for_a_second_public_update(self):
        raw_config = {"kt_num_threads": 32}
        with patch.dict(os.environ, {"ACCELERATE_USE_KT": "false"}):
            args = TrainingArguments(output_dir=tempfile.mkdtemp(), kt_config=raw_config)

        self.assertEqual(args.kt_config, raw_config)
        resolved = {**args.kt_config, "kt_lora_rank": 8, "kt_activation_policy": {"cpu": "retain", "gpu": "recompute"}}
        args.update_kt_config(resolved, adapter_name_or_path="/tmp/adapter")

        self.assertEqual(args.kt_config, resolved)
        self.assertIs(args.accelerator_config.kt_config, args.kt_config)
        self.assertEqual(args.kt_adapter_name_or_path, "/tmp/adapter")

    def test_invalid_adapter_path_does_not_replace_existing_config(self):
        args = self.make_args().update_kt_config({"kt_backend": "AMXBF16"})
        previous_config = args.kt_config
        previous_wrapper = args.hf_kt_config

        with self.assertRaisesRegex(TypeError, "adapter_name_or_path"):
            args.update_kt_config({"kt_backend": "auto"}, adapter_name_or_path=object())

        self.assertIs(args.kt_config, previous_config)
        self.assertIs(args.hf_kt_config, previous_wrapper)

    def test_accepts_typed_kt_config_without_mutating_it(self):
        @kt_owned_config
        @dataclass
        class KTConfig:
            kt_backend: str = "AMXBF16"
            kt_lora_rank: int = 8

        config = KTConfig()
        args = self.make_args().update_kt_config(config)

        self.assertEqual(config, KTConfig())
        self.assertIs(args.kt_config, config)
        self.assertIs(args.hf_kt_config.config, config)
        self.assertIs(args.accelerator_config.kt_config, config)
        self.assertEqual(args.kt_config.kt_backend, "AMXBF16")
        self.assertEqual(args.kt_config.kt_lora_rank, 8)

    def test_typed_kt_config_serializes_without_changing_runtime_identity(self):
        @dataclass(frozen=True)
        class KTActivationPolicy:
            cpu: str = "retain"
            gpu: str = "recompute"

        @kt_owned_config
        @dataclass
        class KTConfig:
            kt_backend: str = "AMXBF16"
            kt_lora_rank: int = 8
            kt_activation_policy: KTActivationPolicy = KTActivationPolicy()

        config = KTConfig()
        args = self.make_args().update_kt_config(config)
        expected = {
            "kt_backend": "AMXBF16",
            "kt_lora_rank": 8,
            "kt_activation_policy": {"cpu": "retain", "gpu": "recompute"},
        }

        payload = args.to_dict()
        json_payload = json.loads(args.to_json_string())

        self.assertEqual(payload["kt_config"], expected)
        self.assertEqual(payload["accelerator_config"]["kt_config"], expected)
        self.assertEqual(json_payload["kt_config"], expected)
        self.assertEqual(json_payload["accelerator_config"]["kt_config"], expected)
        self.assertIs(args.kt_config, config)
        self.assertIs(args.hf_kt_config.config, config)
        self.assertIs(args.accelerator_config.kt_config, config)

    def test_wrapper_replace_copies_mapping_and_clears_runtime_metadata(self):
        original = {"kt_backend": "AMXBF16"}
        wrapper = HfTrainerKTConfig(original)
        wrapper.set_runtime_metadata(kt_checkpoint_files=["old.safetensors"])

        replacement = {"kt_backend": "auto"}
        wrapper.replace(replacement)
        replacement["kt_backend"] = "changed"

        self.assertEqual(wrapper.kt_backend, "auto")
        with self.assertRaises(AttributeError):
            _ = wrapper.kt_checkpoint_files

    def test_trainer_builds_plugin_with_enabled_separate_from_kernel_config(self):
        class Plugin:
            def __init__(self, *, enabled, kt_config):
                self.enabled = enabled
                self.kt_config = kt_config

        trainer = object.__new__(Trainer)
        trainer.model = torch.nn.Linear(2, 2)
        trainer.args = SimpleNamespace(
            mixed_precision="bf16",
            deepspeed_plugin=None,
            ddp_find_unused_parameters=None,
            gradient_checkpointing=False,
            ddp_bucket_cap_mb=None,
            ddp_broadcast_buffers=None,
            parallelism_config=None,
            torch_compile_backend=None,
            torch_compile_mode=None,
            accelerator_config=SimpleNamespace(
                kt_config={"enabled": False, "kt_backend": "AMXBF16", "kt_lora_rank": 8}
            ),
        )

        with (
            patch("transformers.trainer.is_accelerate_available", return_value=False),
            patch("transformers.trainer._accelerate_supports_kt_config", True),
            patch("transformers.trainer.KTransformersPlugin", Plugin),
        ):
            plugin = trainer._build_accelerator_args()["kt_config"]

        self.assertFalse(plugin.enabled)
        self.assertEqual(plugin.kt_config, {"kt_backend": "AMXBF16", "kt_lora_rank": 8})

    def test_trainer_forwards_validated_typed_config_to_plugin_by_identity(self):
        @kt_owned_config
        @dataclass
        class KTConfig:
            kt_backend: str = "AMXBF16"

        class Plugin:
            def __init__(self, *, enabled, kt_config):
                self.enabled = enabled
                self.kt_config = kt_config

        typed_config = KTConfig()
        hf_kt_config = configure_kt(typed_config)
        trainer = object.__new__(Trainer)
        trainer.model = torch.nn.Linear(2, 2)
        trainer.args = SimpleNamespace(
            mixed_precision="bf16",
            deepspeed_plugin=None,
            ddp_find_unused_parameters=None,
            gradient_checkpointing=False,
            ddp_bucket_cap_mb=None,
            ddp_broadcast_buffers=None,
            parallelism_config=None,
            torch_compile_backend=None,
            torch_compile_mode=None,
            hf_kt_config=hf_kt_config,
            accelerator_config=SimpleNamespace(kt_config=typed_config),
        )

        with (
            patch("transformers.trainer.is_accelerate_available", return_value=False),
            patch("transformers.trainer._accelerate_supports_kt_config", True),
            patch("transformers.trainer.KTransformersPlugin", Plugin),
        ):
            plugin = trainer._build_accelerator_args()["kt_config"]

        self.assertTrue(plugin.enabled)
        self.assertIs(plugin.kt_config, typed_config)


if __name__ == "__main__":
    unittest.main()
