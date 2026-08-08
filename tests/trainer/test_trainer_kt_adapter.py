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
from unittest.mock import Mock, patch

import torch

from transformers.trainer import Trainer, _load_fresh_kt_adapter


class _StagedAccelerator:
    def __init__(self, model, events):
        self.model = model
        self.events = events
        self.parallelism_config = None
        self.state = SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=2))

    def register_fsdp2_rank_local_parameters(self, model, parameter_names):
        self.events.append(("register_rank_local", model, tuple(parameter_names)))

    def prepare(self, value):
        self.events.append("prepare_model" if isinstance(value, torch.nn.Module) else "prepare_optimizer")
        return value

    def unwrap_model(self, _model, keep_torch_compile=False):
        return self.model


class TrainerKTAdapterTest(unittest.TestCase):
    def test_fsdp2_save_uses_public_adapter_only_state_api(self):
        trainer = object.__new__(Trainer)
        trainer.args = SimpleNamespace(
            output_dir="/tmp/kt-adapter",
            push_to_hub=False,
            should_save=True,
            world_size=1,
        )
        trainer.is_fsdp_enabled = True
        trainer.is_kt_enabled = True
        trainer.model = torch.nn.Linear(2, 2)
        adapter_state = {"base_model.model.lora_A.default.weight": torch.ones(2, 2)}
        trainer.accelerator = SimpleNamespace(
            get_state_dict=Mock(return_value=adapter_state),
            state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=2, state_dict_type="FULL_STATE_DICT")),
        )
        trainer._kt_placeholder_names = ("base_model.model.layers.0.mlp.experts.gate_up_proj",)
        trainer._save = Mock()

        with (
            patch("transformers.trainer.is_torch_xla_available", return_value=False),
            patch("transformers.trainer.is_sagemaker_mp_enabled", return_value=False),
            patch("transformers.trainer._is_peft_model", return_value=True),
        ):
            trainer.save_model()

        trainer.accelerator.get_state_dict.assert_called_once_with(
            trainer.model,
            adapter_only=True,
            excluded_parameter_names=trainer._kt_placeholder_names,
        )
        trainer._save.assert_called_once_with("/tmp/kt-adapter", state_dict=adapter_state)

    def test_fresh_adapter_uses_explicit_public_path(self):
        model = object()
        with patch("transformers.integrations.kt_artifacts.load_kt_adapter_artifacts") as load_adapter:
            loaded_path = _load_fresh_kt_adapter(model, Path("/tmp/adapter"))

        self.assertEqual(loaded_path, "/tmp/adapter")
        load_adapter.assert_called_once_with(model, "/tmp/adapter")

    def test_missing_fresh_adapter_path_is_a_noop(self):
        with patch("transformers.integrations.kt_artifacts.load_kt_adapter_artifacts") as load_adapter:
            loaded_path = _load_fresh_kt_adapter(object(), None)

        self.assertIsNone(loaded_path)
        load_adapter.assert_not_called()

    def test_staged_lifecycle_adapts_before_optimizer_and_scheduler(self):
        events = []
        model = torch.nn.Linear(2, 2)
        kt_parameter = torch.nn.Parameter(torch.ones(2, 2))
        named_parameters = (("kt.layers.0.experts.fused_lora.gate_lora_a", kt_parameter),)
        adaptation = SimpleNamespace(named_optimizer_parameters=named_parameters, placeholder_names=("placeholder",))
        trainer = object.__new__(Trainer)
        trainer.is_deepspeed_enabled = False
        trainer.is_fsdp_xla_enabled = False
        trainer.is_fsdp_enabled = False
        trainer._created_lr_scheduler = False
        trainer.model = trainer.model_wrapped = model
        trainer.optimizer = None
        trainer.lr_scheduler = None
        trainer.args = SimpleNamespace(kt_adapter_name_or_path="/tmp/adapter")
        trainer.accelerator = _StagedAccelerator(model, events)
        trainer.callback_handler = SimpleNamespace(
            model=None, optimizer=None, lr_scheduler=None, train_dataloader=None
        )
        trainer._wrap_model = Mock(return_value=model)

        def create_optimizer(_model):
            events.append("create_optimizer")
            trainer.optimizer = torch.optim.SGD([model.weight, kt_parameter], lr=0.1)

        def create_scheduler(num_training_steps):
            events.append("create_scheduler")
            trainer.lr_scheduler = object()

        trainer.create_optimizer = create_optimizer
        trainer.create_scheduler = create_scheduler
        trainer._load_kt_adapter_collectively = Mock(side_effect=lambda *_args: events.append("fresh_load"))

        with (
            patch("transformers.trainer.is_sagemaker_mp_enabled", return_value=False),
            patch(
                "transformers.trainer.kt_adapt_peft_lora",
                side_effect=lambda _model: (events.append("adapt"), adaptation)[1],
            ),
            patch("transformers.trainer.get_kt_named_trainable_params", return_value=list(named_parameters)),
        ):
            trainer._prepare_kt_for_training(10, object(), None)

        self.assertEqual(
            events,
            [
                "prepare_model",
                "adapt",
                "fresh_load",
                "create_optimizer",
                "prepare_optimizer",
                "create_scheduler",
            ],
        )
        self.assertEqual(trainer._kt_placeholder_names, ("placeholder",))
        trainer._load_kt_adapter_collectively.assert_called_once_with(
            model,
            "/tmp/adapter",
            "fresh adapter load",
        )

    def test_fsdp2_registers_pre_adaptation_rank_local_names_before_prepare(self):
        events = []
        model = torch.nn.Linear(2, 2)
        trainer = object.__new__(Trainer)
        trainer.is_deepspeed_enabled = False
        trainer.is_fsdp_xla_enabled = False
        trainer.is_fsdp_enabled = True
        trainer._created_lr_scheduler = False
        trainer.model = trainer.model_wrapped = model
        trainer.optimizer = None
        trainer.lr_scheduler = None
        trainer.args = SimpleNamespace(kt_adapter_name_or_path=None)
        trainer.accelerator = _StagedAccelerator(model, events)
        trainer.callback_handler = SimpleNamespace(
            model=None, optimizer=None, lr_scheduler=None, train_dataloader=None
        )
        trainer._wrap_model = Mock(return_value=model)

        def create_optimizer(_model):
            events.append("create_optimizer")
            trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        trainer.create_optimizer = create_optimizer
        trainer.create_scheduler = Mock(side_effect=lambda **_kwargs: events.append("create_scheduler"))

        with (
            patch("transformers.trainer.is_sagemaker_mp_enabled", return_value=False),
            patch("transformers.trainer._is_peft_model", return_value=False),
            patch(
                "transformers.trainer.get_kt_rank_local_parameter_names",
                return_value=("model.layers.0.mlp.experts.gate_up_proj",),
            ),
            patch(
                "transformers.trainer.kt_adapt_peft_lora",
                side_effect=lambda _model: SimpleNamespace(
                    named_optimizer_parameters=(),
                    placeholder_names=("model.layers.0.mlp.experts.gate_up_proj",),
                ),
            ),
            patch("transformers.trainer.get_kt_named_trainable_params", return_value=[]),
        ):
            trainer._prepare_kt_for_training(10, object(), None)

        self.assertEqual(
            events[:3],
            [
                ("register_rank_local", model, ("model.layers.0.mlp.experts.gate_up_proj",)),
                "prepare_model",
                "create_optimizer",
            ],
        )

    def test_fsdp2_resume_restores_standard_then_fused_then_optimizer_state(self):
        events = []
        model = torch.nn.Linear(2, 2)
        trainer = object.__new__(Trainer)
        trainer.is_deepspeed_enabled = False
        trainer.is_fsdp_xla_enabled = False
        trainer.is_fsdp_enabled = True
        trainer._created_lr_scheduler = False
        trainer.model = trainer.model_wrapped = model
        trainer.optimizer = None
        trainer.lr_scheduler = None
        trainer.args = SimpleNamespace(kt_adapter_name_or_path=None)
        trainer.accelerator = _StagedAccelerator(model, events)
        trainer.callback_handler = SimpleNamespace(
            model=None, optimizer=None, lr_scheduler=None, train_dataloader=None
        )
        trainer._wrap_model = Mock(return_value=model)
        trainer._load_from_checkpoint = Mock(side_effect=lambda *_args: events.append("load_standard_adapter"))
        trainer._load_kt_adapter_collectively = Mock(side_effect=lambda *_args: events.append("load_fused_adapter"))
        trainer._load_optimizer_and_scheduler = Mock(side_effect=lambda *_args: events.append("load_optimizer"))
        trainer._load_scaler = Mock(side_effect=lambda *_args: events.append("load_scaler"))

        def create_optimizer(_model):
            events.append("create_optimizer")
            trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def create_scheduler(**_kwargs):
            events.append("create_scheduler")
            trainer.lr_scheduler = object()

        trainer.create_optimizer = create_optimizer
        trainer.create_scheduler = create_scheduler

        with (
            patch("transformers.trainer.is_sagemaker_mp_enabled", return_value=False),
            patch("transformers.trainer._is_peft_model", return_value=False),
            patch("transformers.trainer.get_kt_rank_local_parameter_names", return_value=()),
            patch(
                "transformers.trainer.kt_adapt_peft_lora",
                side_effect=lambda _model: (events.append("adapt"), SimpleNamespace(placeholder_names=()))[1],
            ),
            patch("transformers.trainer.get_kt_named_trainable_params", return_value=[]),
        ):
            trainer._prepare_kt_for_training(10, object(), "/tmp/checkpoint-3")

        self.assertEqual(
            events,
            [
                ("register_rank_local", model, ()),
                "prepare_model",
                "load_standard_adapter",
                "adapt",
                "load_fused_adapter",
                "create_optimizer",
                "prepare_optimizer",
                "create_scheduler",
                "load_optimizer",
                "load_scaler",
            ],
        )
        trainer._load_kt_adapter_collectively.assert_called_once_with(
            model,
            "/tmp/checkpoint-3",
            "resumed adapter load",
        )

    def test_collective_adapter_load_reports_local_failure_before_barrier(self):
        trainer = object.__new__(Trainer)
        trainer._raise_if_kt_checkpoint_failed = Mock()
        trainer._kt_checkpoint_barrier = Mock()
        failure = ValueError("invalid KT adapter")

        with patch(
            "transformers.integrations.kt_artifacts.load_kt_adapter_artifacts",
            side_effect=failure,
        ):
            trainer._load_kt_adapter_collectively(object(), "/tmp/adapter", "fresh adapter load")

        trainer._raise_if_kt_checkpoint_failed.assert_called_once_with(failure, "fresh adapter load")
        trainer._kt_checkpoint_barrier.assert_called_once_with()

    def test_create_optimizer_groups_unregistered_kt_matrix_weights_before_construction(self):
        model = torch.nn.Linear(2, 2)
        kt_parameter = torch.nn.Parameter(torch.ones(2, 2))
        trainer = object.__new__(Trainer)
        trainer.model = model
        trainer.optimizer = None
        trainer.optimizer_cls_and_kwargs = (torch.optim.AdamW, {"lr": 1e-3})
        trainer.args = SimpleNamespace(weight_decay=0.1)
        trainer._kt_optimizer_named_parameters = (("kt.layers.0.experts.fused_lora.gate_lora_a", kt_parameter),)
        trainer.get_decay_parameter_names = lambda _model: ["weight"]

        optimizer = trainer.create_optimizer()

        self.assertTrue(any(parameter is kt_parameter for parameter in optimizer.param_groups[0]["params"]))
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.1)


if __name__ == "__main__":
    unittest.main()
