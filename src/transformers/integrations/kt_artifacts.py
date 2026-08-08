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

"""Lazy bridge to the artifact contract owned by :mod:`kt_kernel.sft.artifacts`.

Transformers coordinates model and Trainer lifecycle only. Artifact schemas, integrity checks, expert layouts, and
atomic publication are implemented by KTransformers.
"""

import contextlib
from typing import Any

from .kt import _get_kt_config


def _artifacts_api():
    try:
        from kt_kernel.sft import artifacts
    except ImportError as error:
        raise ImportError(
            "KTransformers artifact support requires a kt-kernel build exposing `kt_kernel.sft.artifacts`."
        ) from error
    return artifacts


def _active_artifacts_api():
    """Resolve KT's artifact API only while routed experts are KT-owned."""
    from .kt import is_kt_expert_loading_enabled

    if not is_kt_expert_loading_enabled():
        return None
    return _artifacts_api()


def resolve_kt_pretrained_artifacts(
    source_model_name_or_path: str,
    explicit_quantization_config: Any | None,
):
    """Resolve KT-owned model artifacts for the active public KT configuration."""
    kt_config = _get_kt_config()
    if kt_config is None or not getattr(kt_config, "enabled", True):
        return None
    return _artifacts_api().resolve_kt_pretrained_artifacts(
        kt_config,
        source_model_name_or_path,
        explicit_quantization_config,
    )


def validate_kt_pretrained_load(plan: Any, loading_info: Any, model: Any) -> None:
    if plan is not None:
        _artifacts_api().validate_kt_pretrained_load(plan, loading_info, model)


def claim_kt_routed_expert_subtrees(model: Any) -> tuple[str, ...]:
    """Let KT claim its routed-expert subtrees before device-map inference."""
    artifacts = _active_artifacts_api()
    if artifacts is None:
        return ()
    return artifacts.claim_kt_routed_expert_subtrees(model)


def mark_kt_int8_routed_expert_base_parameters(model: Any, plan: Any) -> tuple[str, ...]:
    return _artifacts_api().mark_kt_int8_routed_expert_base_parameters(model, plan)


def is_kt_int8_routed_expert_base_parameter(parameter: Any) -> bool:
    try:
        artifacts = _artifacts_api()
    except ImportError:
        return False
    return artifacts.is_kt_int8_routed_expert_base_parameter(parameter)


@contextlib.contextmanager
def project_kt_routed_experts_out_of_device_map(model: Any):
    artifacts = _active_artifacts_api()
    if artifacts is None:
        yield
        return
    with artifacts.project_kt_routed_experts_out_of_device_map(model):
        yield


def prepare_kt_non_expert_device_map(model: Any, device_map: Any):
    artifacts = _active_artifacts_api()
    if artifacts is None:
        return device_map
    return artifacts.prepare_kt_non_expert_device_map(model, device_map)


@contextlib.contextmanager
def hide_kt_routed_experts_from_dispatch(model: Any):
    artifacts = _active_artifacts_api()
    if artifacts is None:
        yield
        return
    with artifacts.hide_kt_routed_experts_from_dispatch(model):
        yield


@contextlib.contextmanager
def project_kt_int8_routed_experts_out_of_device_map(model: Any):
    try:
        artifacts = _artifacts_api()
    except ImportError:
        yield
        return
    with artifacts.project_kt_int8_routed_experts_out_of_device_map(model):
        yield


def prepare_kt_int8_non_expert_device_map(model: Any, device_map: Any):
    try:
        artifacts = _artifacts_api()
    except ImportError:
        return device_map
    return artifacts.prepare_kt_int8_non_expert_device_map(model, device_map)


@contextlib.contextmanager
def hide_kt_int8_routed_experts_from_dispatch(model: Any):
    try:
        artifacts = _artifacts_api()
    except ImportError:
        yield
        return
    with artifacts.hide_kt_int8_routed_experts_from_dispatch(model):
        yield


def save_kt_adapter_artifacts(model: Any, output_dir: str, *_legacy_callbacks: Any):
    return _artifacts_api().save_kt_adapter_artifacts(model, output_dir)


def load_kt_adapter_artifacts(model: Any, adapter_path: str, *_legacy_callbacks: Any):
    return _artifacts_api().load_kt_adapter_artifacts(model, adapter_path)
