# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING

from ..utils import is_torch_available, strtobool
from ..utils.quantization_config import QuantizationMethod


if TYPE_CHECKING:
    from torch import nn


def is_fsdp_managed_module(module: nn.Module) -> bool:
    if not is_torch_available():
        return False

    import torch

    if not torch.distributed.is_available():
        return False

    import torch.distributed.fsdp

    return isinstance(module, torch.distributed.fsdp.FullyShardedDataParallel) or getattr(
        module, "_is_fsdp_managed_module", False
    )


def is_fsdp_enabled():
    if is_torch_available():
        import torch

        return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
            and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
        )

    return False


def get_fsdp_ckpt_kwargs(excluded_parameter_names=()):
    """
    Returns checkpoint kwargs for FSDP model save and load.

    Selective exclusions require matching support in both Accelerate checkpoint directions.
    """
    from accelerate.utils import load_fsdp_model, save_fsdp_model

    save_parameters = inspect.signature(save_fsdp_model).parameters
    excluded_parameter_names = tuple(excluded_parameter_names)
    if "adapter_only" not in save_parameters:
        if excluded_parameter_names:
            raise RuntimeError("KT FSDP2 checkpoints require Accelerate `adapter_only` checkpoint support.")
        return {}

    kwargs = {"adapter_only": True}
    if excluded_parameter_names:
        load_parameters = inspect.signature(load_fsdp_model).parameters
        if "excluded_parameter_names" not in save_parameters or "excluded_parameter_names" not in load_parameters:
            raise RuntimeError(
                "KT FSDP2 checkpoints require Accelerate save/load support for `excluded_parameter_names`."
            )
        kwargs["excluded_parameter_names"] = excluded_parameter_names
    return kwargs


def update_fsdp_plugin_peft(model, accelerator):
    """
    Updates the FSDP plugin for PEFT LoRA/QLoRA compatibility.

    When using FSDP with PEFT LoRA, the auto wrap policy needs to be updated to additionally wrap
    LoRA trainable layers separately. When using FSDP with QLoRA, the mixed precision policy needs
    to be updated to use the quantization storage data type.
    """
    from peft import PeftConfig
    from peft.utils.other import fsdp_auto_wrap_policy

    if isinstance(model.active_peft_config, PeftConfig):
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
    if (
        getattr(model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES
        and model.hf_quantizer.quantization_config.bnb_4bit_quant_storage.is_floating_point
    ):
        accelerator.state.fsdp_plugin.set_mixed_precision(
            model.hf_quantizer.quantization_config.bnb_4bit_quant_storage, override=True
        )
