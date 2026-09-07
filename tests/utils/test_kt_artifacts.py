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

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from transformers.integrations.accelerate import _get_device_map, accelerate_dispatch
from transformers.integrations.kt import HfTrainerKTConfig, unset_kt_config
from transformers.integrations.kt_artifacts import (
    claim_kt_routed_expert_subtrees,
    hide_kt_routed_experts_from_dispatch,
    load_kt_adapter_artifacts,
    mark_kt_int8_routed_expert_base_parameters,
    prepare_kt_non_expert_device_map,
    prepare_kt_pretrained_config,
    project_kt_routed_experts_out_of_device_map,
    resolve_kt_pretrained_artifacts,
    save_kt_adapter_artifacts,
    validate_kt_pretrained_load,
)


class KTArtifactBridgeTest(unittest.TestCase):
    def tearDown(self):
        unset_kt_config()

    def test_resolve_forwards_the_public_config_without_owning_a_schema(self):
        config = HfTrainerKTConfig({"enabled": True, "kt_expert_weight_format": "int8"})
        plan = object()
        api = SimpleNamespace(resolve_kt_pretrained_artifacts=Mock(return_value=plan))

        with patch("transformers.integrations.kt_artifacts._artifacts_api", return_value=api):
            actual = resolve_kt_pretrained_artifacts("/models/base", None)

        self.assertIs(actual, plan)
        api.resolve_kt_pretrained_artifacts.assert_called_once_with(config, "/models/base", None)

    def test_disabled_config_does_not_import_kt_artifacts(self):
        HfTrainerKTConfig({"enabled": False})
        with patch("transformers.integrations.kt_artifacts._artifacts_api") as api:
            self.assertIsNone(resolve_kt_pretrained_artifacts("/models/base", None))
        api.assert_not_called()

    def test_prepare_config_delegates_source_quantizer_ownership(self):
        kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "rawint4",
            }
        )
        quantization = {"quant_method": "compressed-tensors"}
        config = SimpleNamespace(quantization_config=quantization)
        api = SimpleNamespace(should_disable_kt_source_quantizer=Mock(return_value=True))

        with patch("transformers.integrations.kt_artifacts._artifacts_api", return_value=api):
            self.assertTrue(prepare_kt_pretrained_config(config))

        self.assertFalse(hasattr(config, "quantization_config"))
        api.should_disable_kt_source_quantizer.assert_called_once_with(kt_config, config, None)

    def test_prepare_config_is_a_noop_without_active_kt_ownership(self):
        quantization = {"quant_method": "compressed-tensors"}
        config = SimpleNamespace(quantization_config=quantization)

        with patch("transformers.integrations.kt_artifacts._artifacts_api") as api:
            self.assertFalse(prepare_kt_pretrained_config(config))

        self.assertIs(config.quantization_config, quantization)
        api.assert_not_called()

    def test_loading_validation_and_marking_are_delegated(self):
        plan, loading_info, model = object(), object(), object()
        api = SimpleNamespace(
            validate_kt_pretrained_load=Mock(),
            mark_kt_int8_routed_expert_base_parameters=Mock(return_value=("model.experts",)),
        )

        with patch("transformers.integrations.kt_artifacts._artifacts_api", return_value=api):
            validate_kt_pretrained_load(plan, loading_info, model)
            names = mark_kt_int8_routed_expert_base_parameters(model, plan)

        self.assertEqual(names, ("model.experts",))
        api.validate_kt_pretrained_load.assert_called_once_with(plan, loading_info, model)
        api.mark_kt_int8_routed_expert_base_parameters.assert_called_once_with(model, plan)

    def test_bf16_claim_without_a_pretrained_load_plan_is_delegated(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "bf16",
            }
        )
        model = object()
        api = SimpleNamespace(
            claim_kt_routed_expert_subtrees=Mock(return_value=("model.layers.0.mlp.experts",)),
        )

        with patch("transformers.integrations.kt_artifacts._artifacts_api", return_value=api):
            names = claim_kt_routed_expert_subtrees(model)

        self.assertEqual(names, ("model.layers.0.mlp.experts",))
        api.claim_kt_routed_expert_subtrees.assert_called_once_with(model)

    def test_generic_bridge_is_a_noop_without_active_kt_ownership(self):
        model = object()
        device_map = {"": "cpu"}

        with (
            patch("transformers.integrations.kt.is_kt_expert_loading_enabled", return_value=False),
            patch("transformers.integrations.kt_artifacts._artifacts_api") as api,
        ):
            self.assertEqual(claim_kt_routed_expert_subtrees(model), ())
            with project_kt_routed_experts_out_of_device_map(model):
                pass
            self.assertIs(prepare_kt_non_expert_device_map(model, device_map), device_map)
            with hide_kt_routed_experts_from_dispatch(model):
                pass

        api.assert_not_called()

    def test_device_contexts_are_delegated(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "bf16",
            }
        )
        events = []

        @contextlib.contextmanager
        def context(label):
            events.append(f"enter:{label}")
            yield
            events.append(f"exit:{label}")

        api = SimpleNamespace(
            project_kt_routed_experts_out_of_device_map=lambda _model: context("project"),
            hide_kt_routed_experts_from_dispatch=lambda _model: context("dispatch"),
        )
        with patch("transformers.integrations.kt_artifacts._artifacts_api", return_value=api):
            with project_kt_routed_experts_out_of_device_map(object()):
                events.append("project")
            with hide_kt_routed_experts_from_dispatch(object()):
                events.append("dispatch")

        self.assertEqual(
            events,
            ["enter:project", "project", "exit:project", "enter:dispatch", "dispatch", "exit:dispatch"],
        )

    def test_device_map_and_dispatch_use_generic_contract_in_order(self):
        events = []
        model = SimpleNamespace(_no_split_modules=[], _skip_keys_device_placement=None)
        inferred_map = {"model.layers.0": 0, "model.layers.0.mlp.experts": "cpu"}
        prepared_map = {"model.layers.0": 0}

        @contextlib.contextmanager
        def project(_model):
            events.append("enter:project")
            yield
            events.append("exit:project")

        def balanced(*args, **kwargs):
            events.append("balanced")
            return {0: 1024, "cpu": 4096}

        def infer(*args, **kwargs):
            events.append("infer")
            return inferred_map

        def prepare(_model, device_map):
            events.append("prepare")
            self.assertIs(device_map, inferred_map)
            return prepared_map

        with (
            patch(
                "transformers.integrations.kt_artifacts.project_kt_routed_experts_out_of_device_map",
                new=project,
            ),
            patch("transformers.integrations.kt_artifacts.prepare_kt_non_expert_device_map", new=prepare),
            patch("transformers.integrations.accelerate.get_balanced_memory", new=balanced),
            patch("transformers.integrations.accelerate.infer_auto_device_map", new=infer),
        ):
            actual = _get_device_map(model, "auto", None, None)

        self.assertIs(actual, prepared_map)
        self.assertEqual(events, ["enter:project", "balanced", "infer", "exit:project", "prepare"])

        events.clear()

        @contextlib.contextmanager
        def hide(_model):
            events.append("enter:hide")
            yield
            events.append("exit:hide")

        def dispatch(dispatched_model, **kwargs):
            events.append("dispatch")
            self.assertIs(dispatched_model, model)
            self.assertIs(kwargs["device_map"], prepared_map)

        with (
            patch("transformers.integrations.kt_artifacts.prepare_kt_non_expert_device_map", new=prepare),
            patch("transformers.integrations.kt_artifacts.hide_kt_routed_experts_from_dispatch", new=hide),
            patch("transformers.integrations.accelerate.dispatch_model", new=dispatch),
            patch("transformers.integrations.accelerate.is_fsdp_enabled", return_value=False),
            patch("transformers.integrations.accelerate.is_deepspeed_zero3_enabled", return_value=False),
        ):
            accelerate_dispatch(model, None, inferred_map, None, None, False)

        self.assertEqual(events, ["prepare", "enter:hide", "dispatch", "exit:hide"])

    def test_adapter_save_and_load_are_owned_by_kt(self):
        model = object()
        save_manifest, load_manifest = object(), object()
        api = SimpleNamespace(
            save_kt_adapter_artifacts=Mock(return_value=save_manifest),
            load_kt_adapter_artifacts=Mock(return_value=load_manifest),
        )

        with patch("transformers.integrations.kt_artifacts._artifacts_api", return_value=api):
            self.assertIs(save_kt_adapter_artifacts(model, "/tmp/adapter"), save_manifest)
            self.assertIs(load_kt_adapter_artifacts(model, "/tmp/adapter"), load_manifest)

        api.save_kt_adapter_artifacts.assert_called_once_with(model, "/tmp/adapter")
        api.load_kt_adapter_artifacts.assert_called_once_with(model, "/tmp/adapter")


if __name__ == "__main__":
    unittest.main()
