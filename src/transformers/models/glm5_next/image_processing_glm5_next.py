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

from typing import Any

from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring
from ..glm46v.image_processing_glm46v import Glm46VImageProcessor, smart_resize


class Glm5NextImageProcessorKwargs(ImagesKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    patch_expand_factor: int
    min_image_tokens: int
    max_image_tokens: int


@auto_docstring
class Glm5NextImageProcessor(Glm46VImageProcessor):
    """GLM-5-Next image processor using the checkpoint's factor-1 patch layout."""

    patch_expand_factor = 1
    min_image_tokens = 16
    max_image_tokens = 8000
    valid_kwargs = Glm5NextImageProcessorKwargs

    def __init__(
        self,
        patch_expand_factor: int = 1,
        min_image_tokens: int = 16,
        max_image_tokens: int = 8000,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        size: dict[str, int] | None = None,
        image_processor_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        if patch_expand_factor != 1:
            raise ValueError(f"GLM-5-Next requires patch_expand_factor=1, got {patch_expand_factor!r}.")
        if image_processor_type not in (None, self.__class__.__name__):
            raise ValueError(
                f"Expected image_processor_type={self.__class__.__name__!r}, got {image_processor_type!r}."
            )
        self._validate_token_limits(min_image_tokens, max_image_tokens)

        self.patch_expand_factor = patch_expand_factor
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        if size is None:
            pixels_per_merged_token = patch_size**2 * merge_size**2
            size = {
                "shortest_edge": min_image_tokens * pixels_per_merged_token,
                "longest_edge": max_image_tokens * pixels_per_merged_token,
            }

        super().__init__(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            size=size,
            **kwargs,
        )

    @staticmethod
    def _validate_token_limits(min_image_tokens: int, max_image_tokens: int) -> None:
        if isinstance(min_image_tokens, bool) or not isinstance(min_image_tokens, int) or min_image_tokens <= 0:
            raise ValueError("min_image_tokens must be a positive integer.")
        if isinstance(max_image_tokens, bool) or not isinstance(max_image_tokens, int):
            raise ValueError("max_image_tokens must be a positive integer.")
        if max_image_tokens < min_image_tokens:
            raise ValueError("max_image_tokens must be greater than or equal to min_image_tokens.")

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Glm5NextImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images,
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        patch_expand_factor: int = 1,
        **kwargs,
    ):
        if patch_expand_factor != 1:
            raise ValueError(f"GLM-5-Next requires patch_expand_factor=1, got {patch_expand_factor!r}.")
        if do_resize:
            grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
            resized_images_grouped = {}
            for shape, stacked_images in grouped_images.items():
                height, width = stacked_images.shape[-2:]
                resized_height, resized_width = smart_resize(
                    num_frames=temporal_patch_size,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size * patch_expand_factor,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                resized_images_grouped[shape] = self.resize(
                    stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            images = reorder_images(resized_images_grouped, grouped_images_index)

        return super()._preprocess(
            images=images,
            do_resize=False,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            disable_grouping=disable_grouping,
            return_tensors=return_tensors,
            **kwargs,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        patch_expand_factor = images_kwargs.get("patch_expand_factor", self.patch_expand_factor)
        size = images_kwargs.get("size", self.size)
        min_pixels = size["shortest_edge"] if isinstance(size, dict) else size.shortest_edge
        max_pixels = size["longest_edge"] if isinstance(size, dict) else size.longest_edge
        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            temporal_factor=self.temporal_patch_size,
            factor=patch_size * merge_size * patch_expand_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return (resized_height // patch_size) * (resized_width // patch_size)


__all__ = ["Glm5NextImageProcessor"]
