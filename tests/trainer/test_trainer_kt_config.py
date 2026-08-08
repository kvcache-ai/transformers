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

        @dataclass
        class KTConfig:
            kt_backend: str = "auto"
            kt_lora_rank: int = 8

        typed_config = KTConfig()
        typed_handle = configure_kt(typed_config)

        self.assertEqual(typed_config, KTConfig())
        self.assertEqual(typed_handle.config, {"kt_backend": "auto", "kt_lora_rank": 8})
        self.assertIs(_get_kt_config(), typed_handle)

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
        @dataclass
        class KTConfig:
            kt_backend: str = "AMXBF16"
            kt_lora_rank: int = 8

        config = KTConfig()
        args = self.make_args().update_kt_config(config)

        self.assertEqual(config, KTConfig())
        self.assertEqual(args.kt_config["kt_backend"], "AMXBF16")
        self.assertEqual(args.kt_config["kt_lora_rank"], 8)

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


if __name__ == "__main__":
    unittest.main()
