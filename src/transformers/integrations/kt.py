# Copyright 2025 The HuggingFace Team. All rights reserved.
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


_ENV_TRUE_VALUES = {"1", "true", "yes"}
_KT_EXPERT_KEY_PATTERN = re.compile(r"\.experts\.(\d+\.|gate_up_proj|down_proj|gate_proj|up_proj)")
_kt_config_weak_ref: weakref.ReferenceType | None = None


class HfTrainerKTConfig:
    """
    Lightweight KTransformers config wrapper (similar in spirit to `HfTrainerDeepSpeedConfig`).

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
        "kt_use_lora_experts": ("ACCELERATE_KT_USE_LORA_EXPERTS", bool),
        "kt_lora_expert_num": ("ACCELERATE_KT_LORA_EXPERT_NUM", int),
        "kt_lora_expert_intermediate_size": ("ACCELERATE_KT_LORA_EXPERT_INTERMEDIATE_SIZE", int),
        "kt_lora_rank": ("ACCELERATE_KT_LORA_RANK", int),
        "kt_lora_alpha": ("ACCELERATE_KT_LORA_ALPHA", float),
        "kt_model_max_length": ("ACCELERATE_KT_MODEL_MAX_LENGTH", int),
        "kt_skip_expert_loading": ("ACCELERATE_KT_SKIP_EXPERT_LOADING", bool),
        "kt_share_backward_bb": ("ACCELERATE_KT_SHARE_BACKWARD_BB", bool),
    }

    def __init__(self, kt_config_dict: Any | None):
        # Keep a reference to the processed config so later updates are reflected here.
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
                    self._kt_config[key] = env_val.lower() in _ENV_TRUE_VALUES
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
            if isinstance(self._kt_config, dict):
                self._kt_config.setdefault("kt_share_cache_pool", True)
            elif getattr(self._kt_config, "kt_share_cache_pool", None) is None:
                self._kt_config.kt_share_cache_pool = True

    @property
    def enabled(self) -> bool:
        enabled = self._get("enabled", None)
        # `None` means "enabled unless explicitly disabled".
        return True if enabled is None else bool(enabled)

    @property
    def kt_weight_path(self) -> str | None:
        return self._get("kt_weight_path", None)

    @property
    def kt_skip_expert_loading(self) -> bool | None:
        # If the user explicitly configured it, respect it.
        explicit = self._get("kt_skip_expert_loading", None)
        if explicit is not None:
            return bool(explicit)
        # Default: when KTransformers is enabled, skip expert loading and expect a later KTransformers wrapper to load
        # experts from `kt_weight_path` or via on-the-fly conversion from checkpoint shards.
        return True if self.enabled else False


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


def should_skip_kt_expert_loading() -> bool:
    kt_config: Any | None = _get_kt_config()
    if kt_config is not None:
        enabled = getattr(kt_config, "enabled", None)
        if enabled is False:
            return False
        skip_loading = getattr(kt_config, "kt_skip_expert_loading", None)
        if skip_loading is not None:
            return bool(skip_loading)
        # Default: if KTransformers is enabled, skip loading expert weights.
        return True

    env_enabled = os.environ.get("ACCELERATE_USE_KT", "").lower() in _ENV_TRUE_VALUES
    if not env_enabled:
        return False
    env_skip = os.environ.get("ACCELERATE_KT_SKIP_EXPERT_LOADING", None)
    if env_skip is not None:
        return env_skip.lower() in _ENV_TRUE_VALUES
    # Default: if KTransformers is enabled, skip expert loading.
    return True


def is_kt_expert_key(key: str) -> bool:
    return _KT_EXPERT_KEY_PATTERN.search(key) is not None


def set_kt_checkpoint_metadata(checkpoint_files=None, sharded_metadata=None) -> Any | None:
    kt_config = _get_kt_config()
    if kt_config is None:
        return None

    kt_config_dict = getattr(kt_config, "_kt_config", None)
    if isinstance(kt_config_dict, dict):
        if checkpoint_files is not None:
            kt_config_dict.setdefault("kt_checkpoint_files", checkpoint_files)
        if sharded_metadata is not None:
            kt_config_dict.setdefault("kt_sharded_metadata", sharded_metadata)

    return kt_config


def filter_kt_expert_state_dict(state_dict: dict) -> dict:
    return {key: value for key, value in state_dict.items() if not is_kt_expert_key(key)}


def move_kt_missing_keys_to_cpu(model, missing_keys: set[str]) -> None:
    import torch
    from torch import nn

    expert_missing = {key for key in missing_keys if is_kt_expert_key(key)}
    if not expert_missing:
        return

    missing_keys -= expert_missing
    for key in expert_missing:
        splits = key.rsplit(".", 1)
        if len(splits) != 2:
            continue
        module_path, param_name = splits
        try:
            module = model.get_submodule(module_path)
        except AttributeError:
            continue
        param = getattr(module, param_name, None)
        if param is not None and param.device == torch.device("meta"):
            tiny_storage = torch.UntypedStorage(1, device="cpu")
            fake_tensor = torch.tensor([], dtype=param.dtype, device="cpu").set_(
                tiny_storage,
                storage_offset=0,
                size=param.shape,
                stride=[0] * len(param.shape),
            )
            setattr(module, param_name, nn.Parameter(fake_tensor, requires_grad=False))
        module._is_hf_initialized = True


def wrap_model_with_kt_kernel(model, kt_config):
    try:
        from kt_kernel.sft import wrap_moe_layers_with_kt_wrapper
    except ImportError as error:
        raise ImportError("Using KTransformers requires `kt-kernel` to be installed.") from error

    wrappers = wrap_moe_layers_with_kt_wrapper(model, kt_config)
    model._kt_wrappers = wrappers
    return wrappers
