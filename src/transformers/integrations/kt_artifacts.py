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
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from safetensors import safe_open

from ..utils import logging
from .kt import _get_kt_config, is_kt_int8_expert_loading_enabled


logger = logging.get_logger(__name__)

KT_NON_EXPERT_MANIFEST_NAME = "kt_non_expert_manifest.json"
KT_ADAPTER_MANIFEST_NAME = "kt_adapter_manifest.json"
FUSED_EXPERT_LORA_NAME = "fused_expert_lora.safetensors"
KT_ARTIFACT_MANIFEST_VERSION = 1
_FUSED_LORA_NAMES = (
    "gate_lora_a",
    "gate_lora_b",
    "up_lora_a",
    "up_lora_b",
    "down_lora_a",
    "down_lora_b",
)
_INT8_MANIFEST_CANDIDATES = (
    "kt-weight-manifest.json",
    "kt_weight_manifest.json",
    "kt-ephemeral-manifest.json",
)
_FP32_ROUTER_BIAS = re.compile(r"^model\.layers\.\d+\.mlp\.gate\.e_score_correction_bias$")
_KT_INT8_ROUTED_EXPERT_PARAMETER_MARKER = "_is_kt_int8_routed_expert_base_parameter"
_KT_INT8_ROUTED_EXPERT_MODULE_MARKER = "_is_kt_int8_routed_expert_base_module"
_KT_INT8_ROUTED_EXPERT_PATHS = "_kt_int8_routed_expert_module_paths"
_DEEPSEEK_V3_NUM_HIDDEN_LAYERS = 61
_DEEPSEEK_V3_FIRST_MOE_LAYER = 3
_KT_NON_EXPERT_CONVERTER_NAME = "llamafactory.prepare-kt-cache"
_KT_NON_EXPERT_FP32_EXCEPTIONS = ["model.layers.*.mlp.gate.e_score_correction_bias"]


@dataclass(frozen=True)
class KTNonExpertCache:
    path: str
    manifest: dict[str, Any]
    checkpoint_files: tuple[str, ...]
    index_path: str

    @property
    def fingerprint(self) -> str:
        return self.manifest["fingerprint"]


def is_kt_int8_routed_expert_base_parameter(parameter: Any) -> bool:
    """Whether a parameter is an unloaded DeepSeek routed-expert base weight owned by KT."""
    return getattr(parameter, _KT_INT8_ROUTED_EXPERT_PARAMETER_MARKER, False) is True


def _canonical_deepseek_v3_routed_expert_paths() -> tuple[str, ...]:
    return tuple(
        f"model.layers.{layer_idx}.mlp.experts"
        for layer_idx in range(_DEEPSEEK_V3_FIRST_MOE_LAYER, _DEEPSEEK_V3_NUM_HIDDEN_LAYERS)
    )


def _validated_kt_int8_routed_expert_modules(model: Any) -> tuple[tuple[str, Any], ...]:
    paths = getattr(model, _KT_INT8_ROUTED_EXPERT_PATHS, ())
    if not paths:
        return ()

    expected_paths = _canonical_deepseek_v3_routed_expert_paths()
    if tuple(paths) != expected_paths:
        raise RuntimeError("KT DeepSeek-V3 routed-expert metadata does not match the canonical contract.")

    modules = []
    for path in expected_paths:
        try:
            module = model.get_submodule(path)
        except (AttributeError, KeyError) as error:
            raise RuntimeError(f"KT DeepSeek-V3 routed-expert subtree `{path}` is no longer registered.") from error
        if getattr(module, _KT_INT8_ROUTED_EXPERT_MODULE_MARKER, False) is not True:
            raise RuntimeError(f"KT DeepSeek-V3 routed-expert subtree `{path}` lost its validated marker.")
        modules.append((path, module))
    return tuple(modules)


@contextlib.contextmanager
def project_kt_int8_routed_experts_out_of_device_map(model: Any):
    """Present KT-owned routed experts as zero-sized meta tensors while inferring a non-expert device map."""
    modules = _validated_kt_int8_routed_expert_modules(model)
    if not modules:
        yield
        return

    import torch

    replacements = []
    try:
        for path, expert_module in modules:
            for module in expert_module.modules():
                for name, parameter in tuple(module._parameters.items()):
                    if parameter is None:
                        continue
                    if parameter.device.type != "meta":
                        raise RuntimeError(
                            "KT DeepSeek-V3 automatic device-map projection must run before weight loading; "
                            f"`{path}.{name}` is on {parameter.device}, not meta."
                        )
                    projected = torch.nn.Parameter(
                        torch.empty(0, dtype=parameter.dtype, device="meta"),
                        requires_grad=parameter.requires_grad,
                    )
                    module._parameters[name] = projected
                    replacements.append((module._parameters, name, parameter))

                for name, buffer in tuple(module._buffers.items()):
                    if buffer is None:
                        continue
                    if buffer.device.type != "meta":
                        raise RuntimeError(
                            "KT DeepSeek-V3 automatic device-map projection must run before weight loading; "
                            f"`{path}.{name}` is on {buffer.device}, not meta."
                        )
                    module._buffers[name] = torch.empty(0, dtype=buffer.dtype, device="meta")
                    replacements.append((module._buffers, name, buffer))

        yield
    finally:
        for registry, name, original in reversed(replacements):
            registry[name] = original


def prepare_kt_int8_non_expert_device_map(model: Any, device_map: Any) -> Any:
    """Remove virtual expert placements and reject host offload of real non-expert tensors."""
    modules = _validated_kt_int8_routed_expert_modules(model)
    if not modules:
        return device_map
    if not isinstance(device_map, dict):
        raise RuntimeError("KT DeepSeek-V3 non-expert placement requires a resolved device-map dictionary.")

    import torch

    expert_paths = tuple(path for path, _ in modules)
    resolved_device_map = {
        name: device
        for name, device in device_map.items()
        if not any(name == path or name.startswith(f"{path}.") for path in expert_paths)
    }
    if not resolved_device_map:
        raise RuntimeError("KT DeepSeek-V3 non-expert device map became empty after removing routed experts.")

    def is_host_offload(device: Any) -> bool:
        if device == "disk":
            return True
        if isinstance(device, int):
            return False
        try:
            return torch.device(device).type in {"cpu", "meta"}
        except (RuntimeError, TypeError):
            return False

    host_entries = {name: device for name, device in resolved_device_map.items() if is_host_offload(device)}
    if host_entries:
        details = ", ".join(f"{name or '<root>'}={device}" for name, device in sorted(host_entries.items()))
        raise RuntimeError(
            "KT DeepSeek-V3 INT8 device-map inference offloaded non-expert modules to CPU/disk. "
            "The current adapter inference path requires every non-expert module on an accelerator because PEFT "
            f"would otherwise remove and rebuild Accelerate hooks. Offloaded entries: {details}."
        )
    return resolved_device_map


@contextlib.contextmanager
def hide_kt_int8_routed_experts_from_dispatch(model: Any):
    """Temporarily unregister KT-owned expert subtrees so parent dispatch hooks cannot move their placeholders."""
    modules = _validated_kt_int8_routed_expert_modules(model)
    if not modules:
        yield
        return

    import torch

    replacements = []
    try:
        for path, expert_module in modules:
            parent_path, child_name = path.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
            if parent._modules.get(child_name) is not expert_module:
                raise RuntimeError(f"KT DeepSeek-V3 routed-expert subtree `{path}` changed before dispatch.")
            empty_module = torch.nn.Module()
            parent._modules[child_name] = empty_module
            replacements.append((parent, child_name, expert_module))

        yield
    finally:
        for parent, child_name, expert_module in reversed(replacements):
            parent._modules[child_name] = expert_module


def mark_kt_int8_routed_expert_base_parameters(model: Any, cache: KTNonExpertCache | None) -> tuple[str, ...]:
    """Validate and mark the native DeepSeek-V3 routed-expert base tensors omitted from a KT cache."""
    if cache is None:
        return ()
    if not is_kt_int8_expert_loading_enabled():
        raise RuntimeError("KT routed-expert base parameters can only be marked during KT INT8 expert loading.")

    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "deepseek_v3":
        return ()

    contract = {
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "first_k_dense_replace": getattr(config, "first_k_dense_replace", None),
        "n_routed_experts": getattr(config, "n_routed_experts", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "moe_intermediate_size": getattr(config, "moe_intermediate_size", None),
    }
    if contract["num_hidden_layers"] != _DEEPSEEK_V3_NUM_HIDDEN_LAYERS:
        raise RuntimeError(
            "KT DeepSeek-V3 INT8 cache requires exactly "
            f"{_DEEPSEEK_V3_NUM_HIDDEN_LAYERS} decoder layers, got {contract['num_hidden_layers']!r}."
        )
    if contract["first_k_dense_replace"] != _DEEPSEEK_V3_FIRST_MOE_LAYER:
        raise RuntimeError(
            "KT DeepSeek-V3 INT8 cache requires the first routed-expert layer at index "
            f"{_DEEPSEEK_V3_FIRST_MOE_LAYER}, got {contract['first_k_dense_replace']!r}."
        )
    for field in ("n_routed_experts", "hidden_size", "moe_intermediate_size"):
        value = contract[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"KT DeepSeek-V3 INT8 cache requires a positive integer `{field}`, got {value!r}.")

    try:
        layers = model.get_submodule("model.layers")
    except (AttributeError, KeyError) as error:
        raise RuntimeError("KT DeepSeek-V3 INT8 cache requires the native `model.layers` module hierarchy.") from error
    if not hasattr(layers, "__len__") or len(layers) != _DEEPSEEK_V3_NUM_HIDDEN_LAYERS:
        raise RuntimeError(
            "KT DeepSeek-V3 INT8 cache requires `model.layers` to contain exactly "
            f"{_DEEPSEEK_V3_NUM_HIDDEN_LAYERS} decoder layers."
        )

    expected_paths = _canonical_deepseek_v3_routed_expert_paths()
    expected_parameter_shapes = {
        "gate_up_proj": (
            contract["n_routed_experts"],
            2 * contract["moe_intermediate_size"],
            contract["hidden_size"],
        ),
        "down_proj": (
            contract["n_routed_experts"],
            contract["hidden_size"],
            contract["moe_intermediate_size"],
        ),
    }
    validated_modules = []
    validated_parameters = []
    for layer_idx in range(_DEEPSEEK_V3_NUM_HIDDEN_LAYERS):
        try:
            mlp = layers[layer_idx].mlp
        except (AttributeError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"KT DeepSeek-V3 INT8 cache is missing the MLP subtree for decoder layer {layer_idx}."
            ) from error
        if layer_idx < _DEEPSEEK_V3_FIRST_MOE_LAYER:
            if hasattr(mlp, "experts"):
                raise RuntimeError(f"KT DeepSeek-V3 INT8 cache found routed experts in dense layer {layer_idx}.")
            continue

        expert_path = f"model.layers.{layer_idx}.mlp.experts"
        try:
            experts = model.get_submodule(expert_path)
        except (AttributeError, KeyError) as error:
            raise RuntimeError(
                f"KT DeepSeek-V3 INT8 cache is missing routed-expert subtree `{expert_path}`."
            ) from error
        parameters = dict(experts.named_parameters(recurse=True))
        if set(parameters) != set(expected_parameter_shapes):
            raise RuntimeError(
                f"KT DeepSeek-V3 INT8 cache requires `{expert_path}` parameters "
                f"{sorted(expected_parameter_shapes)}, got {sorted(parameters)}."
            )
        child_modules = dict(experts.named_children())
        if set(child_modules) != {"act_fn"}:
            raise RuntimeError(
                f"KT DeepSeek-V3 INT8 cache requires `{expert_path}` child modules ['act_fn'], "
                f"got {sorted(child_modules)}."
            )
        buffers = dict(experts.named_buffers(recurse=True))
        if buffers:
            raise RuntimeError(
                f"KT DeepSeek-V3 INT8 cache requires `{expert_path}` to have no buffers, got {sorted(buffers)}."
            )
        for name, expected_shape in expected_parameter_shapes.items():
            parameter = parameters[name]
            if tuple(parameter.shape) != expected_shape:
                raise RuntimeError(
                    f"KT DeepSeek-V3 INT8 cache requires `{expert_path}.{name}` shape "
                    f"{expected_shape}, got {tuple(parameter.shape)}."
                )
            validated_parameters.append(parameter)
        validated_modules.append(experts)

    actual_paths = {name for name, _ in model.named_modules() if name.endswith(".mlp.experts")}
    if actual_paths != set(expected_paths):
        raise RuntimeError(
            "KT DeepSeek-V3 INT8 cache routed-expert module paths do not match the canonical contract: "
            f"missing={sorted(set(expected_paths) - actual_paths)}, "
            f"unexpected={sorted(actual_paths - set(expected_paths))}."
        )
    expected_parameter_names = {
        f"{path}.{parameter_name}" for path in expected_paths for parameter_name in expected_parameter_shapes
    }
    actual_parameter_names = {name for name, _ in model.named_parameters() if ".mlp.experts." in name}
    if actual_parameter_names != expected_parameter_names:
        raise RuntimeError(
            "KT DeepSeek-V3 INT8 cache routed-expert parameter names do not match the canonical contract: "
            f"missing={sorted(expected_parameter_names - actual_parameter_names)}, "
            f"unexpected={sorted(actual_parameter_names - expected_parameter_names)}."
        )

    for module in validated_modules:
        setattr(module, _KT_INT8_ROUTED_EXPERT_MODULE_MARKER, True)
    for parameter in validated_parameters:
        setattr(parameter, _KT_INT8_ROUTED_EXPERT_PARAMETER_MARKER, True)
    setattr(model, _KT_INT8_ROUTED_EXPERT_PATHS, expected_paths)
    logger.info(f"Marked {len(expected_paths)} DeepSeek-V3 routed-expert base modules as KT INT8 CPU-owned.")
    return expected_paths


def _sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: str | os.PathLike, description: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {description} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description.capitalize()} {path} must contain a JSON object.")
    return payload


def _require_nonempty_string(value: Any, field: str, manifest_path: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{manifest_path}: `{field}` must be a non-empty string.")
    return value


def _require_sha256(value: Any, field: str, manifest_path: str) -> str:
    digest = _require_nonempty_string(value, field, manifest_path)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"{manifest_path}: `{field}` must be a lowercase SHA256 digest.")
    return digest


def _same_model_source(expected: str | os.PathLike, actual: str) -> bool:
    expected = os.fspath(expected)
    if os.path.exists(expected) or os.path.exists(actual):
        return os.path.realpath(expected) == os.path.realpath(actual)
    return expected == actual


def _compute_kt_non_expert_cache_fingerprint(
    source_fingerprint: str,
    file_records: list[dict[str, Any]],
    tensor_count: int,
    tensor_bytes: int,
    dtype_counts: dict[str, int],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"kt-non-expert-cache-v{KT_ARTIFACT_MANIFEST_VERSION}\0".encode())
    digest.update(source_fingerprint.encode())
    digest.update(b"\0")
    digest.update(str(tensor_count).encode())
    digest.update(b"\0")
    digest.update(str(tensor_bytes).encode())
    digest.update(b"\0")
    digest.update(json.dumps(dtype_counts, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0")
    for record in sorted(file_records, key=lambda item: item["name"]):
        digest.update(record["name"].encode())
        digest.update(b"\0")
        digest.update(str(record["size"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_kt_non_expert_cache(
    cache_path: str | os.PathLike, source_model_name_or_path: str | os.PathLike
) -> KTNonExpertCache:
    """Validate an immutable BF16 non-expert cache before it replaces the checkpoint weight source."""
    cache_path = os.path.abspath(os.fspath(cache_path))
    if os.path.islink(cache_path) or not os.path.isdir(cache_path):
        raise RuntimeError(f"KT non-expert cache must be a real local directory, got {cache_path}.")

    manifest_path = os.path.join(cache_path, KT_NON_EXPERT_MANIFEST_NAME)
    manifest = _read_json_object(manifest_path, "KT non-expert cache manifest")
    if manifest.get("version") != KT_ARTIFACT_MANIFEST_VERSION:
        raise RuntimeError(
            f"{manifest_path}: unsupported version {manifest.get('version')!r}; "
            f"expected {KT_ARTIFACT_MANIFEST_VERSION}."
        )
    if manifest.get("status") != "ready":
        raise RuntimeError(f"{manifest_path}: `status` must be `ready`.")

    fingerprint = _require_sha256(manifest.get("fingerprint"), "fingerprint", manifest_path)
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise RuntimeError(f"{manifest_path}: `source` must be an object.")
    source_path = _require_nonempty_string(
        source.get("model_name_or_path"), "source.model_name_or_path", manifest_path
    )
    source_fingerprint = _require_sha256(source.get("fingerprint"), "source.fingerprint", manifest_path)
    if not _same_model_source(source_model_name_or_path, source_path):
        raise RuntimeError(
            f"{manifest_path}: cache source {source_path!r} does not match requested base "
            f"{os.fspath(source_model_name_or_path)!r}."
        )

    source_artifacts = {
        "config_sha256": os.path.join(source_path, "config.json"),
        "index_sha256": os.path.join(source_path, "model.safetensors.index.json"),
    }
    for field, path in source_artifacts.items():
        expected_digest = _require_sha256(source.get(field), f"source.{field}", manifest_path)
        if os.path.islink(path) or not os.path.isfile(path):
            raise RuntimeError(f"{manifest_path}: source artifact must be a regular non-symlink file: {path}.")
        if _sha256_file(path) != expected_digest:
            raise RuntimeError(f"{manifest_path}: source.{field} does not match {path}.")
    source_config = _read_json_object(source_artifacts["config_sha256"], "source model config")
    source_quantization = source_config.get("quantization_config")
    if not isinstance(source_quantization, dict) or source_quantization.get("quant_method") != "fp8":
        raise RuntimeError(f"{manifest_path}: source config must describe an FP8 checkpoint.")
    source_block_size = source_quantization.get("weight_block_size")
    if (
        not isinstance(source_block_size, (list, tuple))
        or len(source_block_size) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in source_block_size)
    ):
        raise RuntimeError(
            f"{manifest_path}: source config must contain a positive two-element FP8 weight block size."
        )

    converter = manifest.get("converter")
    expected_converter = {
        "name": _KT_NON_EXPERT_CONVERTER_NAME,
        "version": KT_ARTIFACT_MANIFEST_VERSION,
        "default_dtype": "BF16",
        "fp32_exceptions": _KT_NON_EXPERT_FP32_EXCEPTIONS,
        "weight_block_size": list(source_block_size),
    }
    if converter != expected_converter:
        raise RuntimeError(
            f"{manifest_path}: converter contract does not match the supported cache producer; "
            f"expected={expected_converter!r}, got={converter!r}."
        )
    reject_deepseek_mtp = source_config.get("model_type") == "deepseek_v3"

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"{manifest_path}: `files` must be a non-empty list.")

    file_records = {}
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise RuntimeError(f"{manifest_path}: files[{index}] must be an object.")
        name = _require_nonempty_string(record.get("name"), f"files[{index}].name", manifest_path)
        if name != os.path.basename(name) or name in file_records:
            raise RuntimeError(f"{manifest_path}: invalid or duplicate cache filename {name!r}.")
        _require_sha256(record.get("sha256"), f"files[{index}].sha256", manifest_path)
        size = record.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise RuntimeError(f"{manifest_path}: files[{index}].size must be a positive integer.")
        file_records[name] = record

    index_name = "model.safetensors.index.json"
    if index_name not in file_records:
        raise RuntimeError(f"{manifest_path}: sharded cache must list {index_name}.")
    index_path = os.path.join(cache_path, index_name)
    index = _read_json_object(index_path, "safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"{index_path}: `weight_map` must be a non-empty object.")

    shard_names = set()
    for key, shard_name in weight_map.items():
        if not isinstance(key, str) or not key:
            raise RuntimeError(f"{index_path}: tensor names must be non-empty strings.")
        if not isinstance(shard_name, str) or shard_name != os.path.basename(shard_name):
            raise RuntimeError(f"{index_path}: invalid shard name {shard_name!r}.")
        shard_names.add(shard_name)
    if not all(name.endswith(".safetensors") for name in shard_names):
        raise RuntimeError(f"{index_path}: every weight shard must use safetensors.")
    if set(file_records) != shard_names | {index_name}:
        raise RuntimeError(
            f"{manifest_path}: `files` must exactly contain the index and its referenced shards; "
            f"expected {sorted(shard_names | {index_name})}, got {sorted(file_records)}."
        )

    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict):
        raise RuntimeError(f"{manifest_path}: `tensors` must be an object.")
    tensor_count = tensors.get("count")
    tensor_bytes = tensors.get("bytes")
    expected_dtype_counts = tensors.get("dtypes")
    if tensor_count != len(weight_map):
        raise RuntimeError(
            f"{manifest_path}: tensors.count={tensor_count!r} does not match index tensor count {len(weight_map)}."
        )
    if not isinstance(tensor_bytes, int) or isinstance(tensor_bytes, bool) or tensor_bytes <= 0:
        raise RuntimeError(f"{manifest_path}: tensors.bytes must be a positive integer.")
    if (
        not isinstance(expected_dtype_counts, dict)
        or not expected_dtype_counts
        or set(expected_dtype_counts) - {"BF16", "F32"}
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count <= 0
            for count in expected_dtype_counts.values()
        )
        or sum(expected_dtype_counts.values()) != tensor_count
    ):
        raise RuntimeError(
            f"{manifest_path}: tensors.dtypes must contain positive BF16/F32 counts summing to tensors.count."
        )
    expected_fingerprint = _compute_kt_non_expert_cache_fingerprint(
        source_fingerprint,
        list(file_records.values()),
        tensor_count,
        tensor_bytes,
        expected_dtype_counts,
    )
    if fingerprint != expected_fingerprint:
        raise RuntimeError(f"{manifest_path}: fingerprint does not match the source, file, and tensor records.")

    checkpoint_files = []
    observed_keys = set()
    observed_bytes = 0
    observed_dtype_counts: dict[str, int] = {}
    for name, record in file_records.items():
        path = os.path.join(cache_path, name)
        if os.path.islink(path) or not os.path.isfile(path):
            raise RuntimeError(f"{manifest_path}: cache file must be a regular non-symlink file: {path}.")
        actual_size = os.path.getsize(path)
        if actual_size != record["size"]:
            raise RuntimeError(
                f"{manifest_path}: size mismatch for {name}: expected {record['size']}, got {actual_size}."
            )
        actual_digest = _sha256_file(path)
        if actual_digest != record["sha256"]:
            raise RuntimeError(f"{manifest_path}: SHA256 mismatch for {name}.")
        if name == index_name:
            continue

        checkpoint_files.append(path)
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in observed_keys:
                    raise RuntimeError(f"{manifest_path}: duplicate tensor {key!r} across cache shards.")
                tensor_slice = handle.get_slice(key)
                observed_dtype = tensor_slice.get_dtype()
                expected_dtype = "F32" if reject_deepseek_mtp and _FP32_ROUTER_BIAS.fullmatch(key) else "BF16"
                if observed_dtype != expected_dtype:
                    raise RuntimeError(
                        f"{manifest_path}: tensor {key!r} must be {expected_dtype}, got {observed_dtype}."
                    )
                if (
                    ".experts." in key
                    or key.endswith(".weight_scale_inv")
                    or (reject_deepseek_mtp and key.startswith("model.layers.61."))
                ):
                    raise RuntimeError(f"{manifest_path}: cache contains excluded tensor {key!r}.")
                shape = tensor_slice.get_shape()
                elements = 1
                for dimension in shape:
                    elements *= dimension
                observed_bytes += elements * (4 if observed_dtype == "F32" else 2)
                observed_dtype_counts[observed_dtype] = observed_dtype_counts.get(observed_dtype, 0) + 1
                observed_keys.add(key)

    if observed_keys != set(weight_map):
        missing = sorted(set(weight_map) - observed_keys)
        unexpected = sorted(observed_keys - set(weight_map))
        raise RuntimeError(
            f"{manifest_path}: shard/index tensor mismatch: missing={missing}, unexpected={unexpected}."
        )
    if observed_bytes != tensor_bytes:
        raise RuntimeError(
            f"{manifest_path}: tensors.bytes={tensor_bytes} does not match observed bytes {observed_bytes}."
        )
    if observed_dtype_counts != expected_dtype_counts:
        raise RuntimeError(
            f"{manifest_path}: tensors.dtypes={expected_dtype_counts} does not match "
            f"observed dtypes {observed_dtype_counts}."
        )

    logger.info(f"Validated KT non-expert cache {cache_path} ({fingerprint})")
    return KTNonExpertCache(
        path=cache_path,
        manifest=manifest,
        checkpoint_files=tuple(sorted(checkpoint_files)),
        index_path=index_path,
    )


def prepare_kt_non_expert_cache(
    config: Any,
    source_model_name_or_path: str | os.PathLike,
    explicit_quantization_config: Any | None,
) -> KTNonExpertCache | None:
    """Resolve the configured cache and disable the source checkpoint's FP8 quantizer."""
    kt_config = _get_kt_config()
    cache_path = getattr(kt_config, "kt_non_expert_weight_path", None) if kt_config is not None else None
    if cache_path is None:
        cache_path = os.environ.get("ACCELERATE_KT_NON_EXPERT_WEIGHT_PATH")
    if not cache_path:
        return None
    if not is_kt_int8_expert_loading_enabled():
        raise RuntimeError("`kt_non_expert_weight_path` is only supported with KT INT8 expert loading.")
    if explicit_quantization_config is not None:
        raise RuntimeError(
            "`kt_non_expert_weight_path` cannot be combined with an explicit `quantization_config`; "
            "the validated cache already contains BF16 weights."
        )

    cache = validate_kt_non_expert_cache(cache_path, source_model_name_or_path)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    if kt_config is not None and hasattr(kt_config, "_kt_config") and isinstance(kt_config._kt_config, dict):
        kt_config._kt_config["kt_non_expert_cache_manifest"] = cache.manifest
        kt_config._kt_config["kt_non_expert_cache_path"] = cache.path
    return cache


def attach_kt_artifact_provenance(
    model: Any, source_model_name_or_path: str | os.PathLike, cache: KTNonExpertCache | None
) -> None:
    model._kt_base_model_name_or_path = os.fspath(source_model_name_or_path)
    if cache is not None:
        model._kt_non_expert_cache_path = cache.path
        model._kt_non_expert_cache_manifest = cache.manifest


def _find_kt_wrappers(model: Any) -> list[Any]:
    queue = [model]
    visited = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        wrappers = getattr(candidate, "_kt_wrappers", None)
        if wrappers is not None:
            return list(wrappers)
        for attribute in ("base_model", "model", "module"):
            child = getattr(candidate, attribute, None)
            if child is not None and child is not candidate:
                queue.append(child)
    return []


def _torch_dtype_to_safetensors(dtype: Any) -> str:
    mapping = {
        "torch.bfloat16": "BF16",
        "torch.float16": "F16",
        "torch.float32": "F32",
        "torch.float64": "F64",
    }
    dtype_name = str(dtype)
    if dtype_name not in mapping:
        raise RuntimeError(f"Unsupported fused expert LoRA dtype {dtype_name}.")
    return mapping[dtype_name]


def _wrapper_uses_fused_expert_lora(wrapper: Any) -> bool:
    rank = getattr(wrapper, "_lora_rank", 0)
    return bool(
        isinstance(rank, int)
        and rank > 0
        and (
            getattr(wrapper, "_use_fused_expert_lora", False)
            or getattr(wrapper, "_fused_experts", False)
            or getattr(wrapper, "_fused_expert_lora_params", None) is not None
        )
    )


def _expected_fused_lora_contract(model: Any) -> dict[str, dict[str, Any]]:
    contract = {}
    seen_layers = set()
    for wrapper in _find_kt_wrappers(model):
        fused = getattr(wrapper, "_fused_expert_lora_params", None)
        if not fused and not _wrapper_uses_fused_expert_lora(wrapper):
            continue
        layer_idx = getattr(wrapper, "layer_idx", None)
        if not isinstance(layer_idx, int) or isinstance(layer_idx, bool) or layer_idx in seen_layers:
            raise RuntimeError(f"Invalid or duplicate KT wrapper layer index {layer_idx!r}.")
        seen_layers.add(layer_idx)
        if fused:
            if len(fused) != len(_FUSED_LORA_NAMES):
                raise RuntimeError(
                    f"Layer {layer_idx}: expected {len(_FUSED_LORA_NAMES)} fused expert LoRA parameters, "
                    f"got {len(fused)}."
                )
            for name, parameter in zip(_FUSED_LORA_NAMES, fused):
                key = f"layers.{layer_idx}.experts.{name}"
                contract[key] = {
                    "shape": list(parameter.shape),
                    "dtype": _torch_dtype_to_safetensors(parameter.dtype),
                }
            continue

        moe_config = getattr(wrapper, "moe_config", None)
        expert_num = getattr(moe_config, "expert_num", None)
        intermediate_size = getattr(moe_config, "intermediate_size", None)
        hidden_size = getattr(wrapper, "hidden_size", None)
        rank = getattr(wrapper, "_lora_rank", None)
        dimensions = (expert_num, intermediate_size, hidden_size, rank)
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in dimensions):
            raise RuntimeError(f"Layer {layer_idx}: cannot derive the non-owning rank's fused LoRA tensor contract.")
        shapes = (
            (expert_num, rank, hidden_size),
            (expert_num, intermediate_size, rank),
            (expert_num, rank, hidden_size),
            (expert_num, intermediate_size, rank),
            (expert_num, rank, intermediate_size),
            (expert_num, hidden_size, rank),
        )
        for name, shape in zip(_FUSED_LORA_NAMES, shapes):
            contract[f"layers.{layer_idx}.experts.{name}"] = {
                "shape": list(shape),
                "dtype": "BF16",
            }
    return dict(sorted(contract.items()))


def _validate_fused_lora_file(path: str, contract: dict[str, dict[str, Any]]) -> None:
    if os.path.islink(path) or not os.path.isfile(path):
        raise RuntimeError(f"Missing regular {FUSED_EXPERT_LORA_NAME}: {path}.")
    with safe_open(path, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        expected_keys = set(contract)
        if actual_keys != expected_keys:
            raise RuntimeError(
                f"Invalid {FUSED_EXPERT_LORA_NAME} key set: "
                f"missing={sorted(expected_keys - actual_keys)}, unexpected={sorted(actual_keys - expected_keys)}."
            )
        for key, expected in contract.items():
            tensor_slice = handle.get_slice(key)
            if list(tensor_slice.get_shape()) != expected["shape"]:
                raise RuntimeError(
                    f"{key} shape mismatch: expected {expected['shape']}, got {list(tensor_slice.get_shape())}."
                )
            if tensor_slice.get_dtype() != expected["dtype"]:
                raise RuntimeError(
                    f"{key} dtype mismatch: expected {expected['dtype']}, got {tensor_slice.get_dtype()}."
                )


def _get_model_config(model: Any) -> Any | None:
    queue = [model]
    visited = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        config = getattr(candidate, "config", None)
        if config is not None:
            return config
        for attribute in ("base_model", "model", "module"):
            child = getattr(candidate, attribute, None)
            if child is not None and child is not candidate:
                queue.append(child)
    return None


def _load_int8_manifest(weight_path: str) -> tuple[str, dict[str, Any], str]:
    configured_path = None
    kt_config = _get_kt_config()
    if kt_config is not None:
        configured_path = getattr(kt_config, "kt_weight_manifest_path", None)
    candidates = [configured_path] if configured_path else []
    candidates.extend(os.path.join(weight_path, name) for name in _INT8_MANIFEST_CANDIDATES)
    manifest_path = next((path for path in candidates if path and os.path.isfile(path)), None)
    if manifest_path is None:
        raise RuntimeError(f"KT INT8 weight directory {weight_path} has no supported manifest.")
    manifest = _read_json_object(manifest_path, "KT INT8 weight manifest")
    status = manifest.get("status")
    ready = manifest.get("ready")
    if status not in (None, "ready") or ready is False:
        raise RuntimeError(f"KT INT8 weight manifest {manifest_path} is not ready.")
    manifest_sha256 = _sha256_file(manifest_path)
    fingerprint = manifest.get("fingerprint", manifest_sha256)
    _require_nonempty_string(fingerprint, "fingerprint", manifest_path)
    return manifest_path, manifest, fingerprint


def _runtime_adapter_provenance(model: Any) -> dict[str, Any]:
    kt_config = _get_kt_config()
    if kt_config is None:
        raise RuntimeError("KT adapter artifacts require a live KT configuration.")

    cache_manifest = getattr(model, "_kt_non_expert_cache_manifest", None)
    cache_path = getattr(model, "_kt_non_expert_cache_path", None)
    if cache_manifest is None:
        cache_manifest = getattr(kt_config, "kt_non_expert_cache_manifest", None)
        cache_path = getattr(kt_config, "kt_non_expert_cache_path", cache_path)
    if not isinstance(cache_manifest, dict) or not cache_path:
        raise RuntimeError("KT INT8 adapter artifacts require validated non-expert cache provenance.")

    source = cache_manifest.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("Validated KT cache manifest has no source provenance.")
    base_fingerprint = _require_nonempty_string(
        source.get("fingerprint"), "source.fingerprint", KT_NON_EXPERT_MANIFEST_NAME
    )
    cache_fingerprint = _require_nonempty_string(
        cache_manifest.get("fingerprint"), "fingerprint", KT_NON_EXPERT_MANIFEST_NAME
    )
    base_model = getattr(model, "_kt_base_model_name_or_path", None)
    if base_model is None:
        config = _get_model_config(model)
        base_model = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
    if not base_model:
        base_model = source.get("model_name_or_path")

    weight_path = getattr(kt_config, "kt_weight_path", None)
    if not isinstance(weight_path, str) or not weight_path:
        raise RuntimeError("KT INT8 adapter artifacts require `kt_weight_path`.")
    weight_path = os.path.abspath(weight_path)
    int8_manifest_path, _, int8_fingerprint = _load_int8_manifest(weight_path)

    wrappers = [wrapper for wrapper in _find_kt_wrappers(model) if _wrapper_uses_fused_expert_lora(wrapper)]
    ranks = {getattr(wrapper, "_lora_rank", None) for wrapper in wrappers}
    if len(ranks) != 1 or not isinstance(next(iter(ranks), None), int):
        raise RuntimeError(f"KT fused expert wrappers have inconsistent LoRA ranks: {sorted(ranks, key=str)}.")
    rank = next(iter(ranks))
    alpha = getattr(kt_config, "kt_lora_alpha", None)
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise RuntimeError("KT INT8 adapter artifacts require numeric `kt_lora_alpha`.")

    return {
        "base": {
            "model_name_or_path": os.fspath(base_model),
            "fingerprint": base_fingerprint,
        },
        "non_expert_cache": {
            "path": os.path.abspath(os.fspath(cache_path)),
            "fingerprint": cache_fingerprint,
        },
        "int8_experts": {
            "path": weight_path,
            "manifest": os.path.basename(int8_manifest_path),
            "manifest_sha256": _sha256_file(int8_manifest_path),
            "fingerprint": int8_fingerprint,
        },
        "lora": {"rank": rank, "alpha": float(alpha)},
    }


def _file_record(path: str, *, tensor_contract: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    record = {
        "size": os.path.getsize(path),
        "sha256": _sha256_file(path),
    }
    if tensor_contract is not None:
        record["tensor_count"] = len(tensor_contract)
        record["tensors"] = tensor_contract
    return record


def _fsync_file(path: str) -> None:
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_write(payload: dict[str, Any], destination: str) -> None:
    directory = os.path.dirname(destination)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(destination)}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(directory)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(temporary_path)
        raise


def save_kt_adapter_artifacts(
    model: Any,
    output_dir: str | os.PathLike,
    save_kt_moe_to_adapter: Callable[[Any, str], None],
) -> dict[str, Any] | None:
    """Stage KT-owned adapter output and publish a manifest only after atomic file promotion."""
    contract = _expected_fused_lora_contract(model)
    output_dir = os.path.abspath(os.fspath(output_dir))
    if not contract:
        if is_kt_int8_expert_loading_enabled():
            raise RuntimeError(
                "KT INT8 adapter save requires a non-empty fused expert LoRA contract; "
                "the KT wrappers are missing or were not adapted."
            )
        save_kt_moe_to_adapter(model, output_dir)
        return None
    if not is_kt_int8_expert_loading_enabled():
        save_kt_moe_to_adapter(model, output_dir)
        return None

    os.makedirs(output_dir, exist_ok=True)
    adapter_config_path = os.path.join(output_dir, "adapter_config.json")
    standard_adapter_names = [
        name
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if os.path.isfile(os.path.join(output_dir, name))
    ]
    if os.path.islink(adapter_config_path) or not os.path.isfile(adapter_config_path):
        raise RuntimeError(f"KT INT8 adapter save requires a regular {adapter_config_path}.")
    if len(standard_adapter_names) != 1:
        raise RuntimeError(
            "KT INT8 adapter save requires exactly one standard PEFT adapter weight file; "
            f"found {standard_adapter_names}."
        )

    staging_dir = tempfile.mkdtemp(dir=output_dir, prefix=".kt-adapter-stage.")
    try:
        for name in standard_adapter_names:
            shutil.copy2(os.path.join(output_dir, name), os.path.join(staging_dir, name))

        save_kt_moe_to_adapter(model, staging_dir)
        fused_path = os.path.join(staging_dir, FUSED_EXPERT_LORA_NAME)
        _validate_fused_lora_file(fused_path, contract)

        staged_files = []
        for entry in os.scandir(staging_dir):
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise RuntimeError(f"KT adapter staging produced unsupported entry {entry.path}.")
            _fsync_file(entry.path)
            staged_files.append(entry.name)
        if FUSED_EXPERT_LORA_NAME not in staged_files:
            raise RuntimeError(f"KT adapter staging did not produce {FUSED_EXPERT_LORA_NAME}.")

        provenance = _runtime_adapter_provenance(model)
        artifact_records = {
            name: _file_record(
                os.path.join(staging_dir, name),
                tensor_contract=contract if name == FUSED_EXPERT_LORA_NAME else None,
            )
            for name in sorted(staged_files)
        }

        for name in sorted(staged_files):
            os.replace(os.path.join(staging_dir, name), os.path.join(output_dir, name))
        _fsync_directory(output_dir)

        artifact_records["adapter_config.json"] = _file_record(adapter_config_path)

        manifest = {
            "version": KT_ARTIFACT_MANIFEST_VERSION,
            "status": "ready",
            **provenance,
            "artifacts": artifact_records,
        }
        _atomic_json_write(manifest, os.path.join(output_dir, KT_ADAPTER_MANIFEST_NAME))
        logger.info(
            f"Published {KT_ADAPTER_MANIFEST_NAME} with {len(contract)} fused expert LoRA tensors in {output_dir}"
        )
        return manifest
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _validate_artifact_records(adapter_path: str, artifacts: Any) -> None:
    if not isinstance(artifacts, dict) or FUSED_EXPERT_LORA_NAME not in artifacts:
        raise RuntimeError(f"{KT_ADAPTER_MANIFEST_NAME}: `artifacts` must include {FUSED_EXPERT_LORA_NAME}.")
    for name, record in artifacts.items():
        if name != os.path.basename(name) or not isinstance(record, dict):
            raise RuntimeError(f"{KT_ADAPTER_MANIFEST_NAME}: invalid artifact record {name!r}.")
        path = os.path.join(adapter_path, name)
        if os.path.islink(path) or not os.path.isfile(path):
            raise RuntimeError(f"{KT_ADAPTER_MANIFEST_NAME}: missing regular artifact {path}.")
        if record.get("size") != os.path.getsize(path):
            raise RuntimeError(f"{KT_ADAPTER_MANIFEST_NAME}: size mismatch for {name}.")
        expected_digest = record.get("sha256")
        if not isinstance(expected_digest, str) or _sha256_file(path) != expected_digest:
            raise RuntimeError(f"{KT_ADAPTER_MANIFEST_NAME}: SHA256 mismatch for {name}.")
    standard_adapter_names = {name for name in artifacts if name in ("adapter_model.safetensors", "adapter_model.bin")}
    if standard_adapter_names not in ({"adapter_model.safetensors"}, {"adapter_model.bin"}):
        raise RuntimeError(
            f"{KT_ADAPTER_MANIFEST_NAME}: expected exactly one standard PEFT weight artifact, "
            f"got {sorted(standard_adapter_names)}."
        )
    if "adapter_config.json" not in artifacts:
        raise RuntimeError(f"{KT_ADAPTER_MANIFEST_NAME}: missing adapter_config.json artifact.")


def validate_kt_adapter_manifest(model: Any, adapter_path: str | os.PathLike) -> dict[str, Any]:
    adapter_path = os.path.abspath(os.fspath(adapter_path))
    manifest_path = os.path.join(adapter_path, KT_ADAPTER_MANIFEST_NAME)
    manifest = _read_json_object(manifest_path, "KT adapter manifest")
    if manifest.get("version") != KT_ARTIFACT_MANIFEST_VERSION or manifest.get("status") != "ready":
        raise RuntimeError(f"{manifest_path}: expected version={KT_ARTIFACT_MANIFEST_VERSION} and status=`ready`.")

    expected_provenance = _runtime_adapter_provenance(model)
    for field in ("base", "non_expert_cache", "int8_experts"):
        saved = manifest.get(field)
        if not isinstance(saved, dict) or saved.get("fingerprint") != expected_provenance[field]["fingerprint"]:
            raise RuntimeError(
                f"{manifest_path}: {field} fingerprint does not match the current runtime; "
                f"saved={saved!r}, current={expected_provenance[field]!r}."
            )
        if field == "int8_experts" and saved.get("manifest_sha256") != expected_provenance[field]["manifest_sha256"]:
            raise RuntimeError(
                f"{manifest_path}: INT8 expert manifest does not match the current runtime; "
                f"saved={saved!r}, current={expected_provenance[field]!r}."
            )
    if manifest.get("lora") != expected_provenance["lora"]:
        raise RuntimeError(
            f"{manifest_path}: LoRA configuration does not match the current runtime; "
            f"saved={manifest.get('lora')!r}, current={expected_provenance['lora']!r}."
        )

    artifacts = manifest.get("artifacts")
    _validate_artifact_records(adapter_path, artifacts)
    contract = _expected_fused_lora_contract(model)
    fused_record = artifacts[FUSED_EXPERT_LORA_NAME]
    if fused_record.get("tensor_count") != len(contract) or fused_record.get("tensors") != contract:
        raise RuntimeError(f"{manifest_path}: fused expert LoRA tensor contract does not match the current model.")
    _validate_fused_lora_file(os.path.join(adapter_path, FUSED_EXPERT_LORA_NAME), contract)
    return manifest


def load_kt_adapter_artifacts(
    model: Any,
    adapter_path: str | os.PathLike,
    load_kt_moe_from_adapter: Callable[[Any, str], None],
) -> dict[str, Any] | None:
    adapter_path = os.path.abspath(os.fspath(adapter_path))
    contract = _expected_fused_lora_contract(model)
    manifest_path = os.path.join(adapter_path, KT_ADAPTER_MANIFEST_NAME)
    manifest = None
    if is_kt_int8_expert_loading_enabled() and not contract:
        raise RuntimeError(
            "KT INT8 adapter load requires a non-empty fused expert LoRA contract; "
            "the KT wrappers are missing or were not adapted."
        )
    if contract and is_kt_int8_expert_loading_enabled():
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"KT INT8 fused adapter is missing {manifest_path}.")
        manifest = validate_kt_adapter_manifest(model, adapter_path)
    elif os.path.isfile(manifest_path):
        manifest = validate_kt_adapter_manifest(model, adapter_path)

    load_kt_moe_from_adapter(model, adapter_path)
    return manifest
