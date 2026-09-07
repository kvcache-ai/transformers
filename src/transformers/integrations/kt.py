# Copyright 2025 the HuggingFace Inc. team.
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

import copy
import json
import os
import weakref
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any


_kt_config_weak_ref: weakref.ReferenceType | None = None
_kt_environment_owner_ref: weakref.ReferenceType | None = None
_KT_CONFIG_MODULE = "kt_kernel.sft.config"
_KT_CONFIG_CLASS = "KTConfig"


def _clear_collected_kt_state(reference: weakref.ReferenceType) -> None:
    """Release only process-global state owned by the collected KT config."""
    global _kt_config_weak_ref, _kt_environment_owner_ref

    if _kt_config_weak_ref is reference:
        _kt_config_weak_ref = None
    if _kt_environment_owner_ref is reference:
        _kt_environment_owner_ref = None
        os.environ.pop("ACCELERATE_USE_KT", None)


class HfTrainerKTConfig:
    """
    Lightweight KT config wrapper (similar in spirit to `HfTrainerDeepSpeedConfig`).

    A weakref of this object is stored in the module globals so model-loading code (e.g. `from_pretrained`) can
    decide whether to skip loading MoE expert weights before a `Trainer`/`Accelerator` exists.

    This object must stay alive for as long as the model loading needs to observe the KT configuration. In practice,
    `TrainingArguments` stores a reference to it on `self.hf_kt_config`.
    """

    def __init__(self, kt_config_dict: Any | None):
        self._kt_config: Any = {}
        self._runtime_metadata: dict[str, Any] = {}
        self.replace(kt_config_dict)

    def replace(self, kt_config: Any | None) -> "HfTrainerKTConfig":
        """Atomically replace the wrapped KT configuration.

        Mapping inputs are copied so defaults and later runtime metadata never mutate the caller's object.
        Typed KT configuration objects stay typed and are treated as immutable by Transformers.
        """
        replacement = copy.deepcopy(dict(kt_config)) if isinstance(kt_config, Mapping) else kt_config
        if replacement is None:
            replacement = {}
        self._kt_config = replacement
        self._runtime_metadata = {}
        set_kt_config(self)
        return self

    def set_runtime_metadata(self, **metadata: Any) -> None:
        """Attach load-session metadata without mutating the public user configuration."""
        self._runtime_metadata.update({key: value for key, value in metadata.items() if value is not None})

    @property
    def config(self) -> Any:
        """Return the currently wrapped public KT configuration."""
        return self._kt_config

    def _get(self, key: str, default: Any = None) -> Any:
        if key in self._runtime_metadata:
            return self._runtime_metadata[key]
        cfg = self._kt_config
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def __getattr__(self, name: str) -> Any:
        # Allow transparent access to all kt_config dict keys via getattr.
        # This is needed because wrap_moe_layers_with_kt_wrapper accesses
        # kt_backend, kt_num_threads, kt_checkpoint_files, etc. via getattr.
        if name.startswith("_"):
            raise AttributeError(name)
        runtime_metadata = self.__dict__.get("_runtime_metadata", {})
        if name in runtime_metadata:
            return runtime_metadata[name]
        cfg = self.__dict__.get("_kt_config", {})
        if isinstance(cfg, dict) and name in cfg:
            return cfg[name]
        if hasattr(cfg, name):
            return getattr(cfg, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    @property
    def enabled(self) -> bool:
        enabled = self._get("enabled", None)
        # `None` means "enabled unless explicitly disabled".
        return True if enabled is None else bool(enabled)

    @property
    def kt_weight_path(self) -> str | None:
        return self._get("kt_weight_path", None)

    @property
    def kt_non_expert_weight_path(self) -> str | None:
        return self._get("kt_non_expert_weight_path", None)

    @property
    def kt_skip_expert_loading(self) -> bool | None:
        # If the user explicitly configured it, respect it.
        explicit = self._get("kt_skip_expert_loading", None)
        if explicit is not None:
            return bool(explicit)
        # Default: when KT is enabled, we skip expert loading and expect a later KT wrapper to load experts
        # from `kt_weight_path` or via on-the-fly conversion from checkpoint shards.
        return bool(self.enabled)


def _is_typed_kt_config(config: Any) -> bool:
    """Recognize KT's public config type without importing the optional package."""

    config_type = type(config)
    return (
        not isinstance(config, type)
        and is_dataclass(config)
        and config_type.__module__ == _KT_CONFIG_MODULE
        and config_type.__name__ == _KT_CONFIG_CLASS
    )


def _serialize_kt_config(config: Any) -> Any:
    """Convert a typed KT config only at a public serialization boundary."""

    if _is_typed_kt_config(config):
        return asdict(config)
    return copy.deepcopy(config)


def _normalize_kt_config(config: Any) -> Any:
    """Normalize public KT inputs without mutating the caller's object."""
    if isinstance(config, str):
        with open(config, encoding="utf-8") as config_file:
            config = json.load(config_file)
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if _is_typed_kt_config(config):
        return config
    received_type = type(config)
    raise TypeError(
        "`config` must be a mapping, JSON path, None, or an instance of "
        f"{_KT_CONFIG_MODULE}.{_KT_CONFIG_CLASS}; got "
        f"{received_type.__module__}.{received_type.__qualname__}."
    )


def configure_kt(config: Any) -> HfTrainerKTConfig:
    """Configure KT for model loading; retain the returned handle until loading finishes."""
    kt_config = HfTrainerKTConfig(_normalize_kt_config(config))
    _set_kt_config_environment(kt_config, kt_config.enabled)
    return kt_config


def set_kt_config(kt_config: Any) -> None:
    global _kt_config_weak_ref
    _kt_config_weak_ref = weakref.ref(kt_config, _clear_collected_kt_state)


def _set_kt_config_environment(kt_config: Any, enabled: bool) -> None:
    """Mirror explicit KT activation to Accelerate without leaking it past its owner."""
    global _kt_environment_owner_ref

    if not enabled:
        _kt_environment_owner_ref = None
        os.environ.pop("ACCELERATE_USE_KT", None)
        return

    environment_was_enabled = os.environ.get("ACCELERATE_USE_KT", "").lower() in ("1", "true", "yes")
    environment_was_owned = _kt_environment_owner_ref is not None
    os.environ["ACCELERATE_USE_KT"] = "true"
    if not environment_was_enabled or environment_was_owned:
        _kt_environment_owner_ref = weakref.ref(kt_config, _clear_collected_kt_state)


def _is_kt_config_environment_owned() -> bool:
    return _kt_environment_owner_ref is not None and _kt_environment_owner_ref() is not None


def unset_kt_config() -> None:
    global _kt_config_weak_ref, _kt_environment_owner_ref
    _kt_config_weak_ref = None
    if _kt_environment_owner_ref is not None:
        _kt_environment_owner_ref = None
        os.environ.pop("ACCELERATE_USE_KT", None)


def _get_kt_config() -> Any | None:
    if _kt_config_weak_ref is None:
        return None
    return _kt_config_weak_ref()


def is_kt_expert_loading_enabled() -> bool:
    kt_config: Any | None = _get_kt_config()
    if kt_config is not None:
        enabled = getattr(kt_config, "enabled", None)
        if enabled is False:
            return False
        skip_loading = getattr(kt_config, "kt_skip_expert_loading", None)
        if skip_loading is not None:
            return bool(skip_loading)
        # Default: if KT is enabled, skip loading expert weights (they will be supplied by KT later).
        return True

    env_enabled = os.environ.get("ACCELERATE_USE_KT", "").lower() in ("1", "true", "yes")
    if not env_enabled:
        return False
    env_skip = os.environ.get("ACCELERATE_KT_SKIP_EXPERT_LOADING", None)
    if env_skip is not None:
        return env_skip.lower() in ("1", "true", "yes")
    # Default: if KT is enabled, skip expert loading.
    return True


def is_kt_supported_moe_model(model: Any) -> bool:
    from kt_kernel.sft.artifacts import is_kt_supported_moe_model as is_supported

    return is_supported(model)


def is_kt_routed_expert_parameter_name(name: str) -> bool:
    from kt_kernel.sft.artifacts import is_kt_routed_expert_parameter_name as is_routed

    return is_routed(name)


def is_kt_fp8_expert_loading_enabled() -> bool:
    """Whether checkpoint routed experts remain in native block-FP8 storage owned by KT."""
    return _get_kt_expert_weight_format() == "fp8" and is_kt_expert_loading_enabled()


def is_kt_prequantized_expert_loading_enabled() -> bool:
    """Whether KT replaces checkpoint routed experts with a supported pre-quantized backend."""
    return _get_kt_expert_weight_format() in {"int8", "fp8", "rawint4"} and is_kt_expert_loading_enabled()


def _get_kt_expert_weight_format() -> str | None:
    kt_config = _get_kt_config()
    weight_format = getattr(kt_config, "kt_expert_weight_format", None) if kt_config is not None else None
    if weight_format is None:
        weight_format = os.environ.get("ACCELERATE_KT_EXPERT_WEIGHT_FORMAT")
    if not isinstance(weight_format, str):
        return None
    return weight_format.strip().lower()


def _validate_kt_prequantized_loading_info(loading_info: Any, model: Any | None = None) -> None:
    """Fail closed when KT skipped routed experts but did not fully populate the non-expert model."""
    if not is_kt_prequantized_expert_loading_enabled():
        return
    from kt_kernel.sft.artifacts import validate_kt_prequantized_loading_info

    validate_kt_prequantized_loading_info(_get_kt_config(), loading_info, model)
