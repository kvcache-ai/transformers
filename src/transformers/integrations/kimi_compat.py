# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Compatibility for the legacy Kimi K2.5 Hub model's training hooks."""

import inspect
import sys
from functools import wraps


def prepare_remote_kimi_model_class(model_class):
    """Use standard checkpoint enable/disable without editing the downloaded model files."""
    if getattr(getattr(model_class, "config_class", None), "model_type", None) != "kimi_k25":
        return model_class
    if getattr(model_class, "_kimi_training_compat", False):
        return model_class

    from ..modeling_layers import GradientCheckpointingLayer

    module = sys.modules[model_class.__module__]
    encoder = getattr(module, "MoonViT3dEncoder", None)
    if encoder is not None:
        initializer = inspect.unwrap(encoder.__init__)
        # Early checkpoints read this attribute without assigning it in __init__.
        if "use_deterministic_attn" not in inspect.signature(initializer).parameters and not hasattr(
            encoder, "use_deterministic_attn"
        ):
            encoder.use_deterministic_attn = False

    language_model = getattr(module, "DeepseekV3ForCausalLM", None)
    if language_model is None:
        return model_class
    text_module = sys.modules[language_model.__module__]
    decoder = text_module.DeepseekV3DecoderLayer
    text_model = text_module.DeepseekV3Model
    original_forward = text_model.forward
    if "gradient_checkpointing" not in inspect.unwrap(original_forward).__code__.co_names:
        if not issubclass(decoder, GradientCheckpointingLayer):
            text_module.DeepseekV3DecoderLayer = type(
                decoder.__name__, (GradientCheckpointingLayer, decoder), {"__module__": decoder.__module__}
            )

        signature = inspect.signature(original_forward)

        @wraps(original_forward)
        def forward(self, *args, **kwargs):
            if self.training and self.gradient_checkpointing:
                # Disable cache before the legacy model attempts to construct it.
                bound = signature.bind(self, *args, **kwargs)
                bound.arguments["use_cache"] = False
                bound.arguments["past_key_values"] = None
                return original_forward(*bound.args, **bound.kwargs)
            return original_forward(self, *args, **kwargs)

        text_model.forward = forward

    model_class.supports_gradient_checkpointing = True
    model_class._kimi_training_compat = True
    return model_class
