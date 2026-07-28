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

import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

from transformers.integrations.accelerate import (
    _get_device_map,
    accelerate_dispatch,
    compute_module_sizes,
    expand_device_map,
)
from transformers.integrations.kt import HfTrainerKTConfig, unset_kt_config
from transformers.integrations.kt_artifacts import (
    FUSED_EXPERT_LORA_NAME,
    KT_ADAPTER_MANIFEST_NAME,
    KT_NON_EXPERT_MANIFEST_NAME,
    _compute_kt_non_expert_cache_fingerprint,
    load_kt_adapter_artifacts,
    mark_kt_int8_routed_expert_base_parameters,
    prepare_kt_non_expert_cache,
    save_kt_adapter_artifacts,
    validate_kt_non_expert_cache,
)
from transformers.modeling_utils import get_total_byte_count


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class _TinyDenseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 2, dtype=torch.bfloat16))


class _TinyRoutedExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.empty(2, 6, 2, dtype=torch.bfloat16))
        self.down_proj = torch.nn.Parameter(torch.empty(2, 2, 3, dtype=torch.bfloat16))
        self.act_fn = torch.nn.SiLU()


class _TinyMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _TinyRoutedExperts()
        self.gate = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.shared_experts = torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)


class _TinyDecoderLayer(torch.nn.Module):
    def __init__(self, routed):
        super().__init__()
        self.mlp = _TinyMoE() if routed else _TinyDenseMLP()
        self.input_layernorm = torch.nn.Parameter(torch.empty(2, dtype=torch.bfloat16))


class _TinyDeepseekForCausalLM(torch.nn.Module):
    def __init__(self, model_type="deepseek_v3"):
        super().__init__()
        self.config = SimpleNamespace(
            model_type=model_type,
            num_hidden_layers=61,
            first_k_dense_replace=3,
            n_routed_experts=2,
            hidden_size=2,
            moe_intermediate_size=3,
        )
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(4, 2, dtype=torch.bfloat16)
        self.model.layers = torch.nn.ModuleList([_TinyDecoderLayer(routed=layer_idx >= 3) for layer_idx in range(61)])
        self.model.norm = torch.nn.Parameter(torch.empty(2, dtype=torch.bfloat16))
        self.all_tied_weights_keys = {}
        self._tp_plan = {}
        self._no_split_modules = []

    def get_parameter_or_buffer(self, name):
        try:
            return self.get_parameter(name)
        except AttributeError:
            return self.get_buffer(name)


class KTNonExpertCacheTest(unittest.TestCase):
    def tearDown(self):
        unset_kt_config()

    def _make_cache(self, root, extra_tensors=None):
        source = root / "base"
        source.mkdir()
        _write_json(
            source / "config.json",
            {
                "model_type": "deepseek_v3",
                "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
            },
        )
        _write_json(source / "model.safetensors.index.json", {"metadata": {}, "weight_map": {}})
        cache = root / "cache"
        cache.mkdir()
        shards = {
            "model-00001-of-00002.safetensors": {"model.embed_tokens.weight": torch.ones(2, 3, dtype=torch.bfloat16)},
            "model-00002-of-00002.safetensors": {"model.norm.weight": torch.ones(3, dtype=torch.bfloat16)},
        }
        shards["model-00002-of-00002.safetensors"].update(extra_tensors or {})
        weight_map = {}
        tensor_bytes = 0
        dtype_counts = {}
        for name, tensors in shards.items():
            save_file(tensors, cache / name)
            for key, tensor in tensors.items():
                weight_map[key] = name
                tensor_bytes += tensor.numel() * tensor.element_size()
                dtype_name = "F32" if tensor.dtype == torch.float32 else "BF16"
                dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

        index_path = cache / "model.safetensors.index.json"
        _write_json(index_path, {"metadata": {"total_size": tensor_bytes}, "weight_map": weight_map})
        files = []
        for path in sorted(cache.iterdir()):
            files.append({"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)})
        source_fingerprint = "a" * 64
        manifest = {
            "version": 1,
            "status": "ready",
            "fingerprint": _compute_kt_non_expert_cache_fingerprint(
                source_fingerprint,
                files,
                len(weight_map),
                tensor_bytes,
                dtype_counts,
            ),
            "source": {
                "model_name_or_path": str(source),
                "fingerprint": source_fingerprint,
                "config_sha256": _sha256(source / "config.json"),
                "index_sha256": _sha256(source / "model.safetensors.index.json"),
            },
            "converter": {
                "name": "llamafactory.prepare-kt-cache",
                "version": 1,
                "default_dtype": "BF16",
                "fp32_exceptions": ["model.layers.*.mlp.gate.e_score_correction_bias"],
                "weight_block_size": [128, 128],
            },
            "files": files,
            "tensors": {"count": len(weight_map), "bytes": tensor_bytes, "dtypes": dtype_counts},
        }
        _write_json(cache / KT_NON_EXPERT_MANIFEST_NAME, manifest)
        return source, cache

    def test_validates_sharded_bf16_cache_and_disables_source_quantizer(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source, cache = self._make_cache(Path(temporary_directory))
            config = SimpleNamespace(quantization_config={"quant_method": "fp8"})
            self.kt_config = HfTrainerKTConfig(
                {
                    "enabled": True,
                    "kt_skip_expert_loading": True,
                    "kt_expert_weight_format": "int8",
                    "kt_non_expert_weight_path": str(cache),
                }
            )

            resolved = prepare_kt_non_expert_cache(config, source, explicit_quantization_config=None)

            self.assertEqual(len(resolved.fingerprint), 64)
            self.assertEqual(len(resolved.checkpoint_files), 2)
            self.assertFalse(hasattr(config, "quantization_config"))
            self.assertEqual(
                self.kt_config._kt_config["kt_non_expert_cache_manifest"]["source"]["fingerprint"],
                "a" * 64,
            )

    def test_preserves_native_deepseek_router_bias_in_fp32(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, cache = self._make_cache(
                root,
                {
                    "model.layers.3.mlp.gate.e_score_correction_bias": torch.ones(
                        4,
                        dtype=torch.float32,
                    )
                },
            )

            resolved = validate_kt_non_expert_cache(cache, source)

            self.assertEqual(resolved.manifest["tensors"]["dtypes"], {"BF16": 2, "F32": 1})

    def test_rejects_fp32_outside_native_router_bias(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, cache = self._make_cache(
                root,
                {"model.layers.3.input_layernorm.weight": torch.ones(4, dtype=torch.float32)},
            )

            with self.assertRaisesRegex(RuntimeError, "must be BF16"):
                validate_kt_non_expert_cache(cache, source)

    def test_rejects_tampered_cache_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source, cache = self._make_cache(Path(temporary_directory))
            shard = cache / "model-00001-of-00002.safetensors"
            with open(shard, "ab") as handle:
                handle.write(b"tampered")

            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                validate_kt_non_expert_cache(cache, source)

    def test_rejects_manifest_fingerprint_not_derived_from_its_records(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source, cache = self._make_cache(Path(temporary_directory))
            manifest_path = cache / KT_NON_EXPERT_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["fingerprint"] = "0" * 64
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(RuntimeError, "fingerprint does not match"):
                validate_kt_non_expert_cache(cache, source)

    def test_rejects_cache_with_a_mismatched_converter_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source, cache = self._make_cache(Path(temporary_directory))
            manifest_path = cache / KT_NON_EXPERT_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["converter"]["weight_block_size"] = [64, 64]
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(RuntimeError, "converter contract does not match"):
                validate_kt_non_expert_cache(cache, source)

    def test_rejects_cache_for_a_different_base(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, cache = self._make_cache(Path(temporary_directory))
            other_source = Path(temporary_directory) / "other-base"
            other_source.mkdir()

            with self.assertRaisesRegex(RuntimeError, "does not match requested base"):
                validate_kt_non_expert_cache(cache, other_source)

    def test_rejects_deepseek_mtp_tensor(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source, cache = self._make_cache(
                Path(temporary_directory),
                {"model.layers.61.self_attn.q_proj.weight": torch.ones(2, 2, dtype=torch.bfloat16)},
            )

            with self.assertRaisesRegex(RuntimeError, r"excluded tensor .*model\.layers\.61"):
                validate_kt_non_expert_cache(cache, source)

    def test_rejects_cache_after_source_config_or_index_changes(self):
        for filename, field in (
            ("config.json", "source.config_sha256"),
            ("model.safetensors.index.json", "source.index_sha256"),
        ):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    source, cache = self._make_cache(Path(temporary_directory))
                    with open(source / filename, "a", encoding="utf-8") as handle:
                        handle.write("\n")

                    with self.assertRaisesRegex(RuntimeError, field):
                        validate_kt_non_expert_cache(cache, source)

    def test_from_pretrained_keeps_config_provenance_but_reads_cache_weights(self):
        from transformers import BertConfig, BertModel

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "base"
            source.mkdir()
            config = BertConfig(
                hidden_size=8,
                intermediate_size=16,
                num_attention_heads=2,
                num_hidden_layers=1,
                vocab_size=16,
                dtype="bfloat16",
            )
            config.save_pretrained(source)
            config_path = source / "config.json"
            source_config = json.loads(config_path.read_text())
            source_config["quantization_config"] = {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
            _write_json(config_path, source_config)
            source_index_path = source / "model.safetensors.index.json"
            _write_json(source_index_path, {"metadata": {}, "weight_map": {}})

            reference = BertModel(config).to(torch.bfloat16)
            for parameter in reference.parameters():
                parameter.data.fill_(0.25)
            cache = root / "cache"
            cache.mkdir()
            shard_name = "model-00001-of-00001.safetensors"
            state_dict = {key: value.contiguous() for key, value in reference.state_dict().items()}
            save_file(state_dict, cache / shard_name)
            index_path = cache / "model.safetensors.index.json"
            tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
            _write_json(
                index_path,
                {
                    "metadata": {"total_size": tensor_bytes},
                    "weight_map": dict.fromkeys(state_dict, shard_name),
                },
            )
            cache_files = []
            for path in sorted(cache.iterdir()):
                cache_files.append({"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)})
            source_fingerprint = "b" * 64
            cache_fingerprint = _compute_kt_non_expert_cache_fingerprint(
                source_fingerprint,
                cache_files,
                len(state_dict),
                tensor_bytes,
                {"BF16": len(state_dict)},
            )
            _write_json(
                cache / KT_NON_EXPERT_MANIFEST_NAME,
                {
                    "version": 1,
                    "status": "ready",
                    "fingerprint": cache_fingerprint,
                    "source": {
                        "model_name_or_path": str(source),
                        "fingerprint": source_fingerprint,
                        "config_sha256": _sha256(config_path),
                        "index_sha256": _sha256(source_index_path),
                    },
                    "converter": {
                        "name": "llamafactory.prepare-kt-cache",
                        "version": 1,
                        "default_dtype": "BF16",
                        "fp32_exceptions": ["model.layers.*.mlp.gate.e_score_correction_bias"],
                        "weight_block_size": [128, 128],
                    },
                    "files": cache_files,
                    "tensors": {
                        "count": len(state_dict),
                        "bytes": tensor_bytes,
                        "dtypes": {"BF16": len(state_dict)},
                    },
                },
            )
            self.kt_config = HfTrainerKTConfig(
                {
                    "enabled": True,
                    "kt_skip_expert_loading": True,
                    "kt_expert_weight_format": "int8",
                    "kt_non_expert_weight_path": str(cache),
                }
            )
            kt_kernel = types.ModuleType("kt_kernel")
            kt_sft = types.ModuleType("kt_kernel.sft")
            kt_sft.wrap_moe_layers_with_kt_wrapper = Mock(return_value=[])
            kt_kernel.sft = kt_sft

            with patch.dict(sys.modules, {"kt_kernel": kt_kernel, "kt_kernel.sft": kt_sft}):
                loaded = BertModel.from_pretrained(source, dtype=torch.bfloat16)

            self.assertEqual(loaded.config.name_or_path, str(source))
            self.assertFalse(hasattr(loaded.config, "quantization_config"))
            self.assertEqual(loaded._kt_non_expert_cache_path, str(cache))
            self.assertTrue(all(torch.all(parameter == 0.25) for parameter in loaded.parameters()))

    def test_routed_expert_markers_exclude_only_base_experts_from_memory_accounting(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "int8",
            }
        )
        model = _TinyDeepseekForCausalLM()
        paths = mark_kt_int8_routed_expert_base_parameters(model, SimpleNamespace())

        self.assertEqual(paths, tuple(f"model.layers.{idx}.mlp.experts" for idx in range(3, 61)))
        expert_bytes = sum(
            parameter.numel() * parameter.element_size()
            for name, parameter in model.named_parameters()
            if ".mlp.experts." in name
        )
        full_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        expected_non_expert_bytes = full_bytes - expert_bytes

        module_sizes, _ = compute_module_sizes(model, only_modules=False)
        self.assertEqual(module_sizes[""], expected_non_expert_bytes)
        self.assertEqual(module_sizes["model.layers.3.mlp.experts"], 0)
        self.assertGreater(module_sizes["model.layers.3.mlp"], 0)

        gpu = torch.device("cuda:0")
        expanded_device_map = dict.fromkeys((name for name, _ in model.named_parameters()), gpu)
        warmup_bytes = get_total_byte_count(model, expanded_device_map)
        self.assertEqual(warmup_bytes[gpu], expected_non_expert_bytes)

    @staticmethod
    def _move_routed_experts_to_meta(model):
        for layer_idx in range(3, 61):
            experts = model.model.layers[layer_idx].mlp.experts
            for name, parameter in tuple(experts._parameters.items()):
                experts._parameters[name] = torch.nn.Parameter(
                    parameter.detach().to("meta"),
                    requires_grad=parameter.requires_grad,
                )

    def test_auto_device_map_projects_experts_but_returns_only_non_expert_gpu_placements(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "int8",
            }
        )
        model = _TinyDeepseekForCausalLM()
        self._move_routed_experts_to_meta(model)
        expected_paths = mark_kt_int8_routed_expert_base_parameters(model, SimpleNamespace())
        original_gate_up = model.model.layers[3].mlp.experts.gate_up_proj

        def inspect_projected_model(projected_model, **kwargs):
            del kwargs
            projected_gate_up = projected_model.model.layers[3].mlp.experts.gate_up_proj
            self.assertIsNot(projected_gate_up, original_gate_up)
            self.assertEqual(projected_gate_up.device.type, "meta")
            self.assertEqual(projected_gate_up.numel(), 0)
            return {0: 1024, 1: 1024, "cpu": 1024}

        def infer_projected_model(projected_model, **kwargs):
            inspect_projected_model(projected_model, **kwargs)
            return {"model.layers.0": 0, "model.layers.3": 1}

        with (
            patch(
                "transformers.integrations.accelerate.get_balanced_memory",
                side_effect=inspect_projected_model,
            ),
            patch(
                "transformers.integrations.accelerate.infer_auto_device_map",
                side_effect=infer_projected_model,
            ),
        ):
            device_map = _get_device_map(
                model,
                "auto",
                max_memory={0: 1024, 1: 1024, "cpu": 1024},
                hf_quantizer=None,
            )

        self.assertEqual(device_map, {"model.layers.0": 0, "model.layers.3": 1})
        self.assertFalse(set(device_map.values()).intersection({"cpu", "disk"}))
        self.assertIs(model.model.layers[3].mlp.experts.gate_up_proj, original_gate_up)
        self.assertEqual(len(expected_paths), 58)
        parameter_device_map = expand_device_map(device_map, [name for name, _ in model.named_parameters()])
        self.assertEqual(parameter_device_map["model.layers.3.mlp.experts.gate_up_proj"], 1)

    def test_device_map_strips_legacy_exact_expert_overrides_and_rejects_non_expert_cpu_spill(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "int8",
            }
        )
        model = _TinyDeepseekForCausalLM()
        expected_paths = mark_kt_int8_routed_expert_base_parameters(model, SimpleNamespace())
        legacy_map = {"": 0, **dict.fromkeys(expected_paths, "cpu")}

        device_map = _get_device_map(model, legacy_map, max_memory=None, hf_quantizer=None)

        self.assertEqual(device_map, {"": 0})
        with self.assertRaisesRegex(RuntimeError, "offloaded non-expert modules"):
            _get_device_map(
                model,
                {"model.embed_tokens": 0, "model.layers": "cpu"},
                max_memory=None,
                hf_quantizer=None,
            )

    def test_dispatch_hides_experts_from_parent_hooks_and_keeps_peft_from_redispatching(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "int8",
            }
        )
        model = _TinyDeepseekForCausalLM()
        expected_paths = mark_kt_int8_routed_expert_base_parameters(model, SimpleNamespace())
        model._skip_keys_device_placement = None
        original_modules = {path: model.get_submodule(path) for path in expected_paths}
        original_mlp_order = tuple(model.model.layers[3].mlp._modules)
        legacy_load_map = {
            "model.layers.0": 0,
            "model.layers.3": 1,
            **dict.fromkeys(expected_paths, "cpu"),
        }

        def emulate_parent_place_submodules(dispatched_model, **kwargs):
            final_map = kwargs["device_map"]
            self.assertEqual(final_map, {"model.layers.0": 0, "model.layers.3": 1})
            self.assertFalse(set(final_map.values()).intersection({"cpu", "disk"}))
            for path, original_module in original_modules.items():
                hidden_module = dispatched_model.get_submodule(path)
                self.assertIsNot(hidden_module, original_module)
                self.assertEqual(list(hidden_module.parameters()), [])

            layer_parameters = dict(dispatched_model.model.layers[3].named_parameters())
            self.assertFalse(any(name.startswith("mlp.experts.") for name in layer_parameters))
            dispatched_model.hf_device_map = dict(final_map)

        with (
            patch("transformers.integrations.accelerate.dispatch_model", side_effect=emulate_parent_place_submodules),
            patch("transformers.integrations.accelerate.is_fsdp_enabled", return_value=False),
            patch("transformers.integrations.accelerate.is_deepspeed_zero3_enabled", return_value=False),
        ):
            accelerate_dispatch(
                model,
                hf_quantizer=None,
                device_map=legacy_load_map,
                offload_folder=None,
                offload_index=None,
                offload_buffers=False,
            )

        for path, original_module in original_modules.items():
            self.assertIs(model.get_submodule(path), original_module)
        self.assertEqual(tuple(model.model.layers[3].mlp._modules), original_mlp_order)
        self.assertFalse(set(model.hf_device_map.values()).intersection({"cpu", "disk"}))

    def test_dispatch_restores_exact_expert_modules_after_failure(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "int8",
            }
        )
        model = _TinyDeepseekForCausalLM()
        expected_paths = mark_kt_int8_routed_expert_base_parameters(model, SimpleNamespace())
        model._skip_keys_device_placement = None
        original_modules = {path: model.get_submodule(path) for path in expected_paths}

        with (
            patch("transformers.integrations.accelerate.dispatch_model", side_effect=RuntimeError("dispatch failed")),
            patch("transformers.integrations.accelerate.is_fsdp_enabled", return_value=False),
            patch("transformers.integrations.accelerate.is_deepspeed_zero3_enabled", return_value=False),
            self.assertRaisesRegex(RuntimeError, "dispatch failed"),
        ):
            accelerate_dispatch(
                model,
                hf_quantizer=None,
                device_map={"": 0},
                offload_folder=None,
                offload_index=None,
                offload_buffers=False,
            )

        for path, original_module in original_modules.items():
            self.assertIs(model.get_submodule(path), original_module)

    def test_routed_expert_contract_rejects_wrong_parameter_shape(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "int8",
            }
        )
        model = _TinyDeepseekForCausalLM()
        model.model.layers[3].mlp.experts.down_proj = torch.nn.Parameter(torch.empty(2, 3, 2, dtype=torch.bfloat16))

        with self.assertRaisesRegex(RuntimeError, r"down_proj.*shape"):
            mark_kt_int8_routed_expert_base_parameters(model, SimpleNamespace())

    def test_non_deepseek_model_memory_accounting_is_unchanged(self):
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "int8",
            }
        )
        model = _TinyDeepseekForCausalLM(model_type="ordinary")
        self.assertEqual(mark_kt_int8_routed_expert_base_parameters(model, SimpleNamespace()), ())

        full_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        module_sizes, _ = compute_module_sizes(model)
        self.assertEqual(module_sizes[""], full_bytes)

        gpu = torch.device("cuda:0")
        expanded_device_map = dict.fromkeys((name for name, _ in model.named_parameters()), gpu)
        warmup_bytes = get_total_byte_count(model, expanded_device_map)
        self.assertEqual(warmup_bytes[gpu], full_bytes)


class KTAdapterArtifactTest(unittest.TestCase):
    def tearDown(self):
        unset_kt_config()

    def _make_model_and_config(self, root, layers=(3,)):
        cache = root / "cache"
        cache.mkdir()
        int8 = root / "int8"
        int8.mkdir()
        _write_json(
            int8 / "kt-weight-manifest.json",
            {"version": 2, "status": "ready", "fingerprint": "int8-fingerprint"},
        )
        cache_manifest = {
            "version": 1,
            "status": "ready",
            "fingerprint": "cache-fingerprint",
            "source": {
                "model_name_or_path": str(root / "base"),
                "fingerprint": "base-fingerprint",
            },
        }
        wrappers = []
        for layer_idx in layers:
            params = [
                torch.nn.Parameter(torch.zeros(2, 2, 3, dtype=torch.bfloat16)),
                torch.nn.Parameter(torch.zeros(2, 4, 2, dtype=torch.bfloat16)),
                torch.nn.Parameter(torch.zeros(2, 2, 3, dtype=torch.bfloat16)),
                torch.nn.Parameter(torch.zeros(2, 4, 2, dtype=torch.bfloat16)),
                torch.nn.Parameter(torch.zeros(2, 2, 4, dtype=torch.bfloat16)),
                torch.nn.Parameter(torch.zeros(2, 3, 2, dtype=torch.bfloat16)),
            ]
            wrappers.append(
                SimpleNamespace(
                    layer_idx=layer_idx,
                    hidden_size=3,
                    moe_config=SimpleNamespace(expert_num=2, intermediate_size=4),
                    _lora_rank=2,
                    _use_fused_expert_lora=True,
                    _fused_expert_lora_params=params,
                )
            )
        model = SimpleNamespace(
            _kt_wrappers=wrappers,
            _kt_base_model_name_or_path=str(root / "base"),
            _kt_non_expert_cache_path=str(cache),
            _kt_non_expert_cache_manifest=cache_manifest,
        )
        self.kt_config = HfTrainerKTConfig(
            {
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_expert_weight_format": "int8",
                "kt_weight_path": str(int8),
                "kt_lora_rank": 2,
                "kt_lora_alpha": 4,
            }
        )
        return model

    @staticmethod
    def _fake_kt_save(model, output_dir):
        names = ("gate_lora_a", "gate_lora_b", "up_lora_a", "up_lora_b", "down_lora_a", "down_lora_b")
        tensors = {}
        for wrapper in model._kt_wrappers:
            for name, parameter in zip(names, wrapper._fused_expert_lora_params):
                tensors[f"layers.{wrapper.layer_idx}.experts.{name}"] = parameter.detach().clone()
        save_file(tensors, os.path.join(output_dir, FUSED_EXPERT_LORA_NAME))

    def test_atomically_publishes_derived_contract_and_strictly_reloads(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = self._make_model_and_config(root, layers=(3, 4))
            output = root / "adapter"
            output.mkdir()
            save_file({"ordinary.lora_A": torch.ones(2, 2)}, output / "adapter_model.safetensors")
            _write_json(output / "adapter_config.json", {"r": 2, "lora_alpha": 4})

            manifest = save_kt_adapter_artifacts(model, output, self._fake_kt_save)

            self.assertEqual(
                manifest["artifacts"][FUSED_EXPERT_LORA_NAME]["tensor_count"],
                12,
            )
            self.assertTrue((output / KT_ADAPTER_MANIFEST_NAME).is_file())
            self.assertFalse(any(path.name.startswith(".kt-adapter-stage.") for path in output.iterdir()))

            load_function = Mock()
            loaded_manifest = load_kt_adapter_artifacts(model, output, load_function)
            self.assertEqual(loaded_manifest, manifest)
            load_function.assert_called_once_with(model, str(output))

            non_owner = SimpleNamespace(
                _kt_wrappers=[
                    SimpleNamespace(
                        layer_idx=wrapper.layer_idx,
                        hidden_size=wrapper.hidden_size,
                        moe_config=wrapper.moe_config,
                        _lora_rank=wrapper._lora_rank,
                        _use_fused_expert_lora=True,
                        _fused_expert_lora_params=[],
                    )
                    for wrapper in model._kt_wrappers
                ],
                _kt_base_model_name_or_path=model._kt_base_model_name_or_path,
                _kt_non_expert_cache_path=model._kt_non_expert_cache_path,
                _kt_non_expert_cache_manifest=model._kt_non_expert_cache_manifest,
            )
            non_owner_loader = Mock()
            load_kt_adapter_artifacts(non_owner, output, non_owner_loader)
            non_owner_loader.assert_called_once_with(non_owner, str(output))

    def test_rejects_tampering_before_calling_kt_loader(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = self._make_model_and_config(root)
            output = root / "adapter"
            output.mkdir()
            save_file({"ordinary.lora_A": torch.ones(2, 2)}, output / "adapter_model.safetensors")
            _write_json(output / "adapter_config.json", {"r": 2, "lora_alpha": 4})
            save_kt_adapter_artifacts(model, output, self._fake_kt_save)
            with open(output / FUSED_EXPERT_LORA_NAME, "ab") as handle:
                handle.write(b"tampered")
            load_function = Mock()

            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                load_kt_adapter_artifacts(model, output, load_function)

            load_function.assert_not_called()

    def test_rejects_changed_int8_manifest_even_when_declared_fingerprint_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = self._make_model_and_config(root)
            output = root / "adapter"
            output.mkdir()
            save_file({"ordinary.lora_A": torch.ones(2, 2)}, output / "adapter_model.safetensors")
            _write_json(output / "adapter_config.json", {"r": 2, "lora_alpha": 4})
            save_kt_adapter_artifacts(model, output, self._fake_kt_save)

            int8_manifest = root / "int8" / "kt-weight-manifest.json"
            payload = json.loads(int8_manifest.read_text(encoding="utf-8"))
            payload["unrelated_revision"] = 2
            _write_json(int8_manifest, payload)
            load_function = Mock()

            with self.assertRaisesRegex(RuntimeError, "INT8 expert manifest does not match"):
                load_kt_adapter_artifacts(model, output, load_function)

            load_function.assert_not_called()

    def test_int8_adapter_artifacts_fail_closed_without_fused_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = self._make_model_and_config(root)
            model._kt_wrappers = []
            output = root / "adapter"
            output.mkdir()
            save_file({"ordinary.lora_A": torch.ones(2, 2)}, output / "adapter_model.safetensors")
            _write_json(output / "adapter_config.json", {"r": 2, "lora_alpha": 4})
            save_function = Mock()
            load_function = Mock()

            with self.assertRaisesRegex(RuntimeError, "requires a non-empty fused expert LoRA contract"):
                save_kt_adapter_artifacts(model, output, save_function)
            with self.assertRaisesRegex(RuntimeError, "requires a non-empty fused expert LoRA contract"):
                load_kt_adapter_artifacts(model, output, load_function)

            save_function.assert_not_called()
            load_function.assert_not_called()

    def test_failed_stage_does_not_publish_partial_fused_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = self._make_model_and_config(root)
            output = root / "adapter"
            output.mkdir()
            save_file({"ordinary.lora_A": torch.ones(2, 2)}, output / "adapter_model.safetensors")
            _write_json(output / "adapter_config.json", {"r": 2, "lora_alpha": 4})

            def fail_after_write(model, output_dir):
                self._fake_kt_save(model, output_dir)
                raise RuntimeError("save failed")

            with self.assertRaisesRegex(RuntimeError, "save failed"):
                save_kt_adapter_artifacts(model, output, fail_after_write)

            self.assertFalse((output / FUSED_EXPERT_LORA_NAME).exists())
            self.assertFalse((output / KT_ADAPTER_MANIFEST_NAME).exists())
            self.assertFalse(any(path.name.startswith(".kt-adapter-stage.") for path in output.iterdir()))


if __name__ == "__main__":
    unittest.main()
