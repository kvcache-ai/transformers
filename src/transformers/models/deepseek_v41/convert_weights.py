# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import torch

from ...core_model_loading import ConversionOps, WeightConverter, WeightRenaming, _IdentityOp


class NativeDequantize(ConversionOps):
    """Apply each native tensor's scale grid before combining per-expert projections."""

    def convert(self, input_dict, source_patterns, target_patterns, model, full_layer_name, **kwargs):
        module, _, name = full_layer_name.rpartition(".")
        destination = getattr(model.get_submodule(module), name, None)
        projections = []
        for key in source_patterns:
            if key not in input_dict or not key.endswith("weight$"):
                continue
            weights = input_dict[key]
            scales = input_dict.get(key.removesuffix("weight$") + "weight_scale_inv$")
            if scales is None:
                projections.append(torch.stack(weights) if ".experts." in key else weights[0])
                continue
            values = []
            for weight, scale in zip(weights, scales, strict=True):
                if weight.dtype == torch.int8:
                    bits = weight.view(torch.uint8)
                    lut = torch.tensor(
                        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                        device=weight.device,
                    )
                    weight = lut[torch.stack((bits & 15, bits >> 4), dim=-1).long()].flatten(-2)
                rows, cols = weight.shape
                sm, sn = scale.shape
                tiled = weight.float().reshape(sm, rows // sm, sn, cols // sn)
                value = (tiled * scale.float()[:, None, :, None]).reshape(rows, cols)
                values.append(value.to(destination.dtype if destination is not None else torch.bfloat16))
            projections.append(torch.stack(values) if ".experts." in key else values[0])
        value = torch.cat(projections, dim=1) if len(projections) > 1 else projections[0]
        return {target_patterns[0]: value}

    @property
    def reverse_op(self):
        return _IdentityOp()


def native_fp8_conversions(conversions):
    # Rename raw checkpoint scales before applying the model's module renames.
    result = [WeightRenaming(source_patterns=r"^(.+)\.scale$", target_patterns=r"\1.weight_scale_inv")]
    for conv in conversions:
        if isinstance(conv, WeightConverter) and all(".experts." in p for p in conv.source_patterns):
            weights = [p + "$" for p in conv._original_source_patterns]
            scales = [p.removesuffix("weight$") + "weight_scale_inv$" for p in weights]
            conv = WeightConverter(weights + scales, conv._original_target_patterns, operations=[NativeDequantize()])
        result.append(conv)
    result.append(WeightConverter(["weight$", "weight_scale_inv$"], "weight", operations=[NativeDequantize()]))
    return result
