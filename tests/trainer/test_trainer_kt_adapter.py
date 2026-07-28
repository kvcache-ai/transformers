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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from transformers.trainer import (
    Trainer,
    _get_kt_fsdp2_peft_state_dict,
    _load_fresh_kt_adapter,
)


class TrainerKTAdapterReloadTest(unittest.TestCase):
    def test_kt_fsdp2_peft_save_avoids_accelerate_full_state_gather(self):
        trainer = object.__new__(Trainer)
        trainer.args = SimpleNamespace(
            output_dir="/tmp/kt-adapter",
            push_to_hub=False,
            should_save=True,
        )
        trainer.is_fsdp_enabled = True
        trainer.is_kt_enabled = True
        trainer.model = torch.nn.Linear(2, 2)
        trainer.accelerator = SimpleNamespace(
            get_state_dict=unittest.mock.Mock(),
            state=SimpleNamespace(
                fsdp_plugin=SimpleNamespace(
                    fsdp_version=2,
                    state_dict_type="FULL_STATE_DICT",
                )
            ),
        )
        trainer._save = unittest.mock.Mock()
        adapter_state = {"base_model.model.lora_A.default.weight": torch.ones(2, 2)}

        with (
            patch("transformers.trainer.is_torch_xla_available", return_value=False),
            patch("transformers.trainer.is_sagemaker_mp_enabled", return_value=False),
            patch("transformers.trainer._is_peft_model", return_value=True),
            patch(
                "transformers.trainer._get_kt_fsdp2_peft_state_dict",
                return_value=adapter_state,
            ) as gather_adapter,
        ):
            trainer.save_model()

        gather_adapter.assert_called_once_with(trainer.model)
        trainer.accelerator.get_state_dict.assert_not_called()
        trainer._save.assert_called_once_with(
            "/tmp/kt-adapter",
            state_dict=adapter_state,
        )

    def test_fsdp2_adapter_save_gathers_only_trainable_state(self):
        model = torch.nn.Linear(2, 2)
        expected = {"base_model.model.lora_A.default.weight": torch.ones(2, 2)}

        with patch(
            "torch.distributed.checkpoint.state_dict.get_model_state_dict",
            return_value=expected,
        ) as get_state_dict:
            actual = _get_kt_fsdp2_peft_state_dict(model)

        self.assertIs(actual, expected)
        get_state_dict.assert_called_once()
        self.assertIs(get_state_dict.call_args.args[0], model)
        options = get_state_dict.call_args.kwargs["options"]
        self.assertTrue(options.full_state_dict)
        self.assertTrue(options.cpu_offload)
        self.assertTrue(options.ignore_frozen_params)

    def test_fsdp2_adapter_save_excludes_frozen_parameters(self):
        model = torch.nn.Linear(2, 2)
        model.weight.requires_grad_(False)

        state_dict = _get_kt_fsdp2_peft_state_dict(model)

        self.assertEqual(set(state_dict), {"bias"})
        self.assertEqual(state_dict["bias"].device.type, "cpu")

    def test_loads_fresh_adapter_after_kt_adaptation(self):
        model = SimpleNamespace(_kt_adapter_path=Path("/tmp/adapter"))

        with patch("transformers.trainer.load_kt_moe_from_adapter") as load_adapter:
            loaded_path = _load_fresh_kt_adapter(model, resume_from_checkpoint=None)

        self.assertEqual(loaded_path, "/tmp/adapter")
        self.assertTrue(model._kt_adapter_loaded)
        load_adapter.assert_called_once_with(model, "/tmp/adapter")

    def test_already_loaded_adapter_is_not_loaded_twice(self):
        model = SimpleNamespace(_kt_adapter_path="/tmp/adapter", _kt_adapter_loaded=True)

        with patch("transformers.trainer.load_kt_moe_from_adapter") as load_adapter:
            loaded_path = _load_fresh_kt_adapter(model, resume_from_checkpoint=None)

        self.assertIsNone(loaded_path)
        load_adapter.assert_not_called()

    def test_checkpoint_resume_does_not_double_load_initial_adapter(self):
        model = SimpleNamespace(_kt_adapter_path="/tmp/initial-adapter")

        with patch("transformers.trainer.load_kt_moe_from_adapter") as load_adapter:
            loaded_path = _load_fresh_kt_adapter(model, resume_from_checkpoint="/tmp/checkpoint-3")

        self.assertIsNone(loaded_path)
        load_adapter.assert_not_called()

    def test_model_without_adapter_path_is_unchanged(self):
        model = SimpleNamespace()

        with patch("transformers.trainer.load_kt_moe_from_adapter") as load_adapter:
            loaded_path = _load_fresh_kt_adapter(model, resume_from_checkpoint=None)

        self.assertIsNone(loaded_path)
        load_adapter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
