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

import math
from typing import Any

import numpy as np

from ...processing_utils import VideosKwargs
from ...video_utils import VideoMetadata
from ..glm46v.video_processing_glm46v import Glm46VVideoProcessor


class Glm5NextVideoProcessorInitKwargs(VideosKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    patch_expand_factor: int
    min_image_tokens: int
    max_image_tokens: int
    max_frames: int
    max_duration: int
    dynamic_fps_thresholds: list[list[float]]


class Glm5NextVideoProcessor(Glm46VVideoProcessor):
    """Official GLM-5-Next temporal sampling on top of GLM-4.6V patchification."""

    patch_expand_factor = 1
    min_image_tokens = 16
    max_image_tokens = 240000
    max_frames = 640
    max_duration = 2400
    fps = 2
    dynamic_fps_thresholds = [[30, 3], [300, 1], [2400, 0.5]]
    valid_kwargs = Glm5NextVideoProcessorInitKwargs

    def __init__(
        self,
        patch_expand_factor: int = 1,
        min_image_tokens: int = 16,
        max_image_tokens: int = 240000,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        fps: int | float | None = 2,
        max_frames: int = 640,
        max_duration: int = 2400,
        dynamic_fps_thresholds: list[list[float]] | None = None,
        size: dict[str, int] | None = None,
        video_processor_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        if patch_expand_factor != 1:
            raise ValueError(f"GLM-5-Next requires patch_expand_factor=1, got {patch_expand_factor!r}.")
        if video_processor_type not in (None, self.__class__.__name__):
            raise ValueError(
                f"Expected video_processor_type={self.__class__.__name__!r}, got {video_processor_type!r}."
            )
        if isinstance(min_image_tokens, bool) or not isinstance(min_image_tokens, int) or min_image_tokens <= 0:
            raise ValueError("min_image_tokens must be a positive integer.")
        if isinstance(max_image_tokens, bool) or not isinstance(max_image_tokens, int):
            raise ValueError("max_image_tokens must be a positive integer.")
        if max_image_tokens < min_image_tokens:
            raise ValueError("max_image_tokens must be greater than or equal to min_image_tokens.")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < temporal_patch_size:
            raise ValueError("max_frames must be an integer no smaller than temporal_patch_size.")
        if isinstance(max_duration, bool) or not isinstance(max_duration, int) or max_duration <= 0:
            raise ValueError("max_duration must be a positive integer.")

        thresholds = dynamic_fps_thresholds or self.dynamic_fps_thresholds
        self._validate_dynamic_fps_thresholds(thresholds, max_duration)
        self.patch_expand_factor = patch_expand_factor
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.max_frames = max_frames
        self.max_duration = max_duration
        self.dynamic_fps_thresholds = thresholds
        if size is None:
            pixels_per_merged_token = temporal_patch_size * patch_size**2 * merge_size**2
            size = {
                "shortest_edge": min_image_tokens * pixels_per_merged_token,
                "longest_edge": max_image_tokens * pixels_per_merged_token,
            }

        super().__init__(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            fps=fps,
            size=size,
            **kwargs,
        )

    @staticmethod
    def _validate_dynamic_fps_thresholds(thresholds: list[list[float]], max_duration: int) -> None:
        if not thresholds or any(len(entry) != 2 for entry in thresholds):
            raise ValueError("dynamic_fps_thresholds must contain [max_duration, fps] pairs.")
        durations = [entry[0] for entry in thresholds]
        if durations != sorted(durations) or durations[-1] != max_duration:
            raise ValueError("dynamic_fps_thresholds must be sorted and end at max_duration.")
        if any(entry[1] <= 0 for entry in thresholds):
            raise ValueError("dynamic_fps_thresholds fps values must be positive.")

    def sample_frames(
        self,
        metadata: VideoMetadata,
        fps: int | float | None = None,
        **kwargs,
    ) -> np.ndarray:
        if metadata is None or metadata.fps is None:
            raise ValueError(
                "GLM-5-Next frame sampling requires video metadata with fps; "
                "pass VideoMetadata or set do_sample_frames=False."
            )
        if metadata.total_num_frames <= 0:
            raise ValueError("GLM-5-Next videos must contain at least one frame.")

        total_frames = metadata.total_num_frames
        max_frame_idx = total_frames - 1
        duration = metadata.duration or round(max_frame_idx / metadata.fps) + 1
        max_seconds = int(duration)
        effective_duration = min(duration, self.max_duration)

        target_fps = fps
        if target_fps is None:
            target_fps = next(
                threshold_fps
                for threshold_duration, threshold_fps in self.dynamic_fps_thresholds
                if effective_duration <= threshold_duration
            )
        if target_fps <= 0:
            raise ValueError("fps must be positive.")
        target_fps *= self.temporal_patch_size

        extract_t = min(max(int(effective_duration * target_fps), self.temporal_patch_size), self.max_frames)
        duration_per_frame = 1 / metadata.fps
        timestamps = [frame_index * duration_per_frame for frame_index in range(total_frames)]

        if total_frames < extract_t:
            frame_indices = [math.floor(index * total_frames / extract_t) for index in range(extract_t)]
        else:
            frame_indices = []
            current_second = 0.0
            inverse_fps = 1 / target_fps
            for frame_index, timestamp in enumerate(timestamps):
                if timestamp >= current_second:
                    current_second += inverse_fps
                    frame_indices.append(frame_index)
                    if current_second >= max_seconds:
                        break

        if len(frame_indices) < extract_t:
            start = frame_indices[0] if frame_indices else 0
            end = frame_indices[-1] if frame_indices else max(total_frames - 1, 0)
            frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
        elif len(frame_indices) > extract_t:
            frame_indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()

        unique_indices = list(dict.fromkeys(frame_indices))
        if len(unique_indices) % self.temporal_patch_size:
            unique_indices.extend(
                [unique_indices[-1]] * (self.temporal_patch_size - len(unique_indices) % self.temporal_patch_size)
            )
        return np.asarray(unique_indices, dtype=np.int64)


__all__ = ["Glm5NextVideoProcessor"]
