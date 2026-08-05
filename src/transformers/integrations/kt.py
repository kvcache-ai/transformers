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

import os
import re
import weakref
from typing import Any


_kt_config_weak_ref: weakref.ReferenceType | None = None


class HfTrainerKTConfig:
    """
    Lightweight KT config wrapper (similar in spirit to `HfTrainerDeepSpeedConfig`).

    A weakref of this object is stored in the module globals so model-loading code (e.g. `from_pretrained`) can
    decide whether to skip loading MoE expert weights before a `Trainer`/`Accelerator` exists.

    This object must stay alive for as long as the model loading needs to observe the KT configuration. In practice,
    `TrainingArguments` stores a reference to it on `self.hf_kt_config`.
    """

    # Mapping from kt_config dict keys to ACCELERATE_KT_* environment variables.
    # Dict keys use kt_ prefix, matching KTConfig field names exactly.
    _ENV_MAPPING: dict[str, tuple[str, type]] = {
        "kt_backend": ("ACCELERATE_KT_BACKEND", str),
        "kt_num_gpu_experts": ("ACCELERATE_KT_NUM_GPU_EXPERTS", int),
        "kt_num_threads": ("ACCELERATE_KT_NUM_THREADS", int),
        "kt_tp_enabled": ("ACCELERATE_KT_TP_ENABLED", bool),
        "kt_threadpool_count": ("ACCELERATE_KT_THREADPOOL_COUNT", int),
        "kt_max_cache_depth": ("ACCELERATE_KT_MAX_CACHE_DEPTH", int),
        "kt_weight_path": ("ACCELERATE_KT_WEIGHT_PATH", str),
        "kt_non_expert_weight_path": ("ACCELERATE_KT_NON_EXPERT_WEIGHT_PATH", str),
        "kt_expert_weight_format": ("ACCELERATE_KT_EXPERT_WEIGHT_FORMAT", str),
        "kt_use_lora_experts": ("ACCELERATE_KT_USE_LORA_EXPERTS", bool),
        "kt_lora_expert_num": ("ACCELERATE_KT_LORA_EXPERT_NUM", int),
        "kt_lora_expert_intermediate_size": ("ACCELERATE_KT_LORA_EXPERT_INTERMEDIATE_SIZE", int),
        "kt_lora_rank": ("ACCELERATE_KT_LORA_RANK", int),
        "kt_lora_alpha": ("ACCELERATE_KT_LORA_ALPHA", float),
        "kt_lora_dropout": ("ACCELERATE_KT_LORA_DROPOUT", float),
        "kt_model_max_length": ("ACCELERATE_KT_MODEL_MAX_LENGTH", int),
        "kt_skip_expert_loading": ("ACCELERATE_KT_SKIP_EXPERT_LOADING", bool),
    }

    def __init__(self, kt_config_dict: Any | None):
        # Keep a reference to the original config so later mutations (e.g. filling defaults) are reflected here.
        self._kt_config = kt_config_dict if kt_config_dict is not None else {}

        # Fill missing config values from ACCELERATE_KT_* env vars.
        # These are set by `accelerate launch --config_file` via _apply_kt_config_to_env()
        # before the training script starts, but kt_config_dict may be None/empty when
        # TrainingArguments.__post_init__ runs (e.g. when kt_config is not passed explicitly
        # and the only signal is the ACCELERATE_USE_KT env var).
        if isinstance(self._kt_config, dict):
            for key, (env_key, typ) in self._ENV_MAPPING.items():
                if key in self._kt_config:
                    continue
                env_val = os.environ.get(env_key)
                if env_val is None or env_val == "":
                    continue
                if typ is bool:
                    self._kt_config[key] = env_val.lower() in ("1", "true", "yes")
                elif typ is int:
                    self._kt_config[key] = int(env_val)
                elif typ is float:
                    self._kt_config[key] = float(env_val)
                else:
                    self._kt_config[key] = env_val

        set_kt_config(self)

    def _get(self, key: str, default: Any = None) -> Any:
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
        cfg = self.__dict__.get("_kt_config", {})
        if isinstance(cfg, dict) and name in cfg:
            return cfg[name]
        if hasattr(cfg, name):
            return getattr(cfg, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def trainer_config_process(self, args):
        """Adjust kt_config with TrainingArguments values, similar to DeepSpeed's trainer_config_process."""
        if getattr(args, "gradient_checkpointing", False):
            self._kt_config.setdefault("kt_share_cache_pool", True)

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


def set_kt_config(kt_config: Any) -> None:
    global _kt_config_weak_ref
    _kt_config_weak_ref = weakref.ref(kt_config)


def unset_kt_config() -> None:
    global _kt_config_weak_ref
    _kt_config_weak_ref = None


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


_KT_ROUTED_EXPERT_KEY = re.compile(r"\.experts\.(?:\d+\.|gate_up_proj|down_proj|gate_proj|up_proj)")
_DEEPSEEK_V3_MTP_KEY = re.compile(r"^model\.layers\.61\.")


def is_kt_routed_expert_parameter_name(name: str) -> bool:
    return _KT_ROUTED_EXPERT_KEY.search(name) is not None


def is_kt_int8_expert_loading_enabled() -> bool:
    """Whether checkpoint expert tensors are replaced by pre-quantized KT INT8 weights."""
    return _get_kt_expert_weight_format() == "int8" and is_kt_expert_loading_enabled()


def is_kt_fp8_expert_loading_enabled() -> bool:
    """Whether checkpoint routed experts remain in native block-FP8 storage owned by KT."""
    return _get_kt_expert_weight_format() == "fp8" and is_kt_expert_loading_enabled()


def is_kt_prequantized_expert_loading_enabled() -> bool:
    """Whether KT replaces checkpoint routed experts with a supported pre-quantized backend."""
    return _get_kt_expert_weight_format() in {"int8", "fp8"} and is_kt_expert_loading_enabled()


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

    weight_format = _get_kt_expert_weight_format()
    format_label = weight_format.upper() if weight_format is not None else "PREQUANTIZED"

    config = getattr(model, "config", None)
    allow_deepseek_mtp = (
        getattr(config, "model_type", None) == "deepseek_v3" and getattr(config, "num_hidden_layers", None) == 61
    )
    missing_keys = sorted(key for key in loading_info.missing_keys if not _KT_ROUTED_EXPERT_KEY.search(key))
    mismatched_keys = sorted(
        mismatch for mismatch in loading_info.mismatched_keys if not _KT_ROUTED_EXPERT_KEY.search(mismatch[0])
    )
    conversion_errors = {
        key: error for key, error in loading_info.conversion_errors.items() if not _KT_ROUTED_EXPERT_KEY.search(key)
    }
    unexpected_keys = sorted(
        key for key in loading_info.unexpected_keys if not (allow_deepseek_mtp and _DEEPSEEK_V3_MTP_KEY.match(key))
    )
    error_msgs = list(loading_info.error_msgs)

    failures = []
    if missing_keys:
        failures.append(f"missing_keys={missing_keys}")
    if mismatched_keys:
        failures.append(f"mismatched_keys={mismatched_keys}")
    if conversion_errors:
        failures.append(f"conversion_errors={conversion_errors}")
    if unexpected_keys:
        failures.append(f"unexpected_keys={unexpected_keys}")
    if error_msgs:
        failures.append(f"error_msgs={error_msgs}")

    if failures:
        raise RuntimeError(
            f"KT {format_label} checkpoint loading requires an exact non-expert model match; "
            + "; ".join(failures)
        )


def _validate_kt_int8_loading_info(loading_info: Any, model: Any | None = None) -> None:
    """Backward-compatible alias for callers that still use the INT8-specific validator name."""
    _validate_kt_prequantized_loading_info(loading_info, model)
