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

import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available
from transformers.video_utils import VideoMetadata


if is_torch_available() and is_torchvision_available():
    from transformers import Glm5NextImageProcessor, Glm5NextProcessor, Glm5NextVideoProcessor


OFFICIAL_IMAGE_CONFIG = {
    "do_rescale": True,
    "patch_expand_factor": 1,
    "merge_size": 2,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "temporal_patch_size": 2,
    "patch_size": 14,
    "min_image_tokens": 16,
    "max_image_tokens": 8000,
    "image_processor_type": "Glm5NextImageProcessor",
}

OFFICIAL_VIDEO_CONFIG = {
    "do_rescale": True,
    "video_processor_type": "Glm5NextVideoProcessor",
    "patch_expand_factor": 1,
    "merge_size": 2,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "temporal_patch_size": 2,
    "patch_size": 14,
    "min_image_tokens": 16,
    "max_image_tokens": 240000,
    "fps": 2,
}


@require_torch
@require_vision
class Glm5NextProcessorTest(unittest.TestCase):
    def test_public_classes_are_exported(self):
        self.assertEqual(Glm5NextProcessor.__name__, "Glm5NextProcessor")
        self.assertEqual(Glm5NextImageProcessor.__name__, "Glm5NextImageProcessor")
        self.assertEqual(Glm5NextVideoProcessor.__name__, "Glm5NextVideoProcessor")

    def test_official_image_config_and_patch_count(self):
        processor = Glm5NextImageProcessor(**OFFICIAL_IMAGE_CONFIG)
        self.assertEqual(processor.patch_expand_factor, 1)
        self.assertEqual(processor.size.shortest_edge, 16 * 14**2 * 2**2)
        self.assertEqual(processor.size.longest_edge, 8000 * 14**2 * 2**2)

        image = np.zeros((56, 84, 3), dtype=np.uint8)
        output = processor(images=image, return_tensors="pt")
        grid = output["image_grid_thw"]
        self.assertEqual(tuple(grid.shape), (1, 3))
        self.assertEqual(output["pixel_values"].shape[0], int(grid.prod().item()))
        self.assertEqual(output["pixel_values"].shape[1], 3 * 2 * 14**2)

    def test_official_video_config_and_patch_count(self):
        processor = Glm5NextVideoProcessor(**OFFICIAL_VIDEO_CONFIG)
        self.assertEqual(processor.patch_expand_factor, 1)
        self.assertEqual(processor.fps, 2)

        video = np.zeros((8, 56, 84, 3), dtype=np.uint8)
        metadata = [VideoMetadata(total_num_frames=8, fps=4, duration=2, frames_indices=list(range(8)))]
        output = processor(
            videos=video,
            video_metadata=metadata,
            do_sample_frames=False,
            return_metadata=True,
            return_tensors="pt",
        )
        grid = output["video_grid_thw"]
        self.assertEqual(tuple(grid.shape), (1, 3))
        self.assertEqual(output["pixel_values_videos"].shape[0], int(grid.prod().item()))
        self.assertEqual(output["pixel_values_videos"].shape[1], 3 * 2 * 14**2)
        self.assertEqual(len(output["video_metadata"]), 1)

    def test_frame_sampling_is_temporally_aligned(self):
        processor = Glm5NextVideoProcessor(**OFFICIAL_VIDEO_CONFIG)
        metadata = VideoMetadata(total_num_frames=120, fps=30, duration=4)
        indices = processor.sample_frames(metadata, fps=2)
        self.assertEqual(len(indices), 16)
        self.assertEqual(len(indices) % processor.temporal_patch_size, 0)
        self.assertTrue(np.all(indices[1:] > indices[:-1]))
        self.assertGreaterEqual(indices[0], 0)
        self.assertLess(indices[-1], metadata.total_num_frames)

    def test_invalid_processor_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "patch_expand_factor=1"):
            Glm5NextImageProcessor(**{**OFFICIAL_IMAGE_CONFIG, "patch_expand_factor": 2})
        with self.assertRaisesRegex(ValueError, "max_image_tokens"):
            Glm5NextVideoProcessor(**{**OFFICIAL_VIDEO_CONFIG, "max_image_tokens": 8})


if __name__ == "__main__":
    unittest.main()
