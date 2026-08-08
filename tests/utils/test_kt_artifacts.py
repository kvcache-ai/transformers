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

from transformers.integrations.kt import HfTrainerKTConfig, unset_kt_config
from transformers.integrations.kt_artifacts import (
    hide_kt_int8_routed_experts_from_dispatch,
    load_kt_adapter_artifacts,
    mark_kt_int8_routed_expert_base_parameters,
    project_kt_int8_routed_experts_out_of_device_map,
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

    def test_device_contexts_are_delegated(self):
        events = []

        @contextlib.contextmanager
        def context(label):
            events.append(f"enter:{label}")
            yield
            events.append(f"exit:{label}")

        api = SimpleNamespace(
            project_kt_int8_routed_experts_out_of_device_map=lambda _model: context("project"),
            hide_kt_int8_routed_experts_from_dispatch=lambda _model: context("dispatch"),
        )
        with patch("transformers.integrations.kt_artifacts._artifacts_api", return_value=api):
            with project_kt_int8_routed_experts_out_of_device_map(object()):
                events.append("project")
            with hide_kt_int8_routed_experts_from_dispatch(object()):
                events.append("dispatch")

        self.assertEqual(
            events,
            ["enter:project", "project", "exit:project", "enter:dispatch", "dispatch", "exit:dispatch"],
        )

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
