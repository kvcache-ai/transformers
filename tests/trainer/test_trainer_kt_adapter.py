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

from transformers.trainer import _load_fresh_kt_adapter


class TrainerKTAdapterReloadTest(unittest.TestCase):
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
