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

import json
import os
import tempfile
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from transformers.testing_utils import require_accelerate, require_torch
from transformers.trainer import (
    KT_OPTIMIZER_INDEX_NAME,
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    Trainer,
    _atomic_path_save,
    _atomic_torch_save,
    _kt_optimizer_rank_files,
    _read_kt_optimizer_manifest,
)


def _make_adamw(group_sizes: list[int], gradient_value: float) -> torch.optim.AdamW:
    groups = []
    for group_index, group_size in enumerate(group_sizes):
        params = [torch.nn.Parameter(torch.full((2,), float(group_index + 1))) for _ in range(group_size)]
        groups.append({"params": params, "weight_decay": 0.1 * group_index})
    optimizer = torch.optim.AdamW(groups, lr=0.01)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.full_like(parameter, gradient_value)
    optimizer.step()
    return optimizer


def _make_trainer(optimizer: torch.optim.Optimizer, rank: int, world_size: int) -> Trainer:
    trainer = object.__new__(Trainer)
    trainer.args = SimpleNamespace(
        device=torch.device("cpu"),
        process_index=rank,
        should_save=rank == 0,
        world_size=world_size,
    )
    trainer.optimizer = optimizer
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer.is_deepspeed_enabled = False
    trainer.is_fsdp_enabled = True
    trainer.is_fsdp_xla_v1_enabled = False
    trainer.is_kt_enabled = True
    return trainer


def _disable_checkpoint_collectives(trainer: Trainer) -> None:
    def raise_local_error(error, operation):
        if error is not None:
            raise RuntimeError(f"{operation}: {error}") from error

    trainer._raise_if_kt_checkpoint_failed = raise_local_error
    trainer._kt_checkpoint_barrier = lambda: None


def _distributed_tail_failure_worker(rank: int, init_file: str, checkpoint: str, operation: str) -> None:
    import transformers.trainer as trainer_module

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    trainer = _make_trainer(_make_adamw([1, 1], float(rank + 1)), rank=rank, world_size=2)
    original_atomic_save = trainer_module._atomic_torch_save

    def injected_atomic_save(state_dict, destination):
        failing_rank = 0 if operation == "scheduler" else 1
        if rank == failing_rank:
            raise OSError(f"injected {operation} failure")
        original_atomic_save(state_dict, destination)

    try:
        with (
            patch("transformers.trainer._atomic_torch_save", side_effect=injected_atomic_save),
            patch("transformers.trainer.torch.cuda.is_available", return_value=False),
        ):
            if operation == "scheduler":
                trainer._save_kt_fsdp_scheduler(checkpoint)
            else:
                trainer._run_kt_checkpoint_io(
                    "RNG state save",
                    partial(trainer._save_rng_state, checkpoint),
                )
    except RuntimeError as error:
        if (operation == "scheduler" and rank == 0) or (operation == "rng" and rank == 1):
            assert "failed on this rank" in str(error)
        else:
            assert "failed on another rank" in str(error)
    else:
        raise AssertionError(f"injected {operation} checkpoint failure did not propagate")
    finally:
        dist.destroy_process_group()


@require_torch
@require_accelerate
class TrainerKTOptimizerCheckpointTest(unittest.TestCase):
    def run_distributed_tail_failure(self, operation: str):
        with tempfile.TemporaryDirectory() as checkpoint:
            init_file = os.path.join(checkpoint, "distributed_init")
            context = torch.multiprocessing.get_context("spawn")
            processes = [
                context.Process(
                    target=_distributed_tail_failure_worker,
                    args=(rank, init_file, checkpoint, operation),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=20)
            hanging_processes = [process for process in processes if process.is_alive()]
            for process in hanging_processes:
                process.terminate()
                process.join()
            self.assertEqual(hanging_processes, [], f"{operation} failure left a distributed rank hanging")
            self.assertEqual([process.exitcode for process in processes], [0, 0])

    def test_two_rank_scheduler_failure_propagates_without_hang(self):
        self.run_distributed_tail_failure("scheduler")

    def test_two_rank_rng_failure_propagates_without_hang(self):
        self.run_distributed_tail_failure("rng")

    def test_two_rank_checkpoint_loads_each_ranks_parameter_layout(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            source_rank_0 = _make_trainer(_make_adamw([1, 2], 1.0), rank=0, world_size=2)
            source_rank_1 = _make_trainer(_make_adamw([2, 1], 2.0), rank=1, world_size=2)
            _disable_checkpoint_collectives(source_rank_0)
            _disable_checkpoint_collectives(source_rank_1)

            # Save rank 1 first because rank 0 publishes the manifest after its local save.
            source_rank_1._save_kt_fsdp_optimizer(checkpoint)
            source_rank_0._save_kt_fsdp_optimizer(checkpoint)
            torch.save(source_rank_0.lr_scheduler.state_dict(), os.path.join(checkpoint, SCHEDULER_NAME))
            source_rank_0._publish_kt_fsdp_optimizer_manifest(checkpoint)

            with open(os.path.join(checkpoint, KT_OPTIMIZER_INDEX_NAME), encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["world_size"], 2)
            self.assertEqual(manifest["rank_files"], _kt_optimizer_rank_files(2))

            restored_rank_0 = _make_trainer(_make_adamw([1, 2], 0.0), rank=0, world_size=2)
            restored_rank_1 = _make_trainer(_make_adamw([2, 1], 0.0), rank=1, world_size=2)
            restored_rank_0._load_optimizer_and_scheduler(checkpoint)
            restored_rank_1._load_optimizer_and_scheduler(checkpoint)

            for state in restored_rank_0.optimizer.state.values():
                torch.testing.assert_close(state["exp_avg"], torch.full_like(state["exp_avg"], 0.1))
            for state in restored_rank_1.optimizer.state.values():
                torch.testing.assert_close(state["exp_avg"], torch.full_like(state["exp_avg"], 0.2))

    def test_single_rank_kt_fsdp_checkpoint_keeps_optimizer_pt_compatibility(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            source = _make_trainer(_make_adamw([1, 1], 3.0), rank=0, world_size=1)
            _atomic_torch_save(source.optimizer.state_dict(), os.path.join(checkpoint, OPTIMIZER_NAME))
            torch.save(source.lr_scheduler.state_dict(), os.path.join(checkpoint, SCHEDULER_NAME))

            restored = _make_trainer(_make_adamw([1, 1], 0.0), rank=0, world_size=1)
            restored._load_optimizer_and_scheduler(checkpoint)
            for state in restored.optimizer.state.values():
                torch.testing.assert_close(state["exp_avg"], torch.full_like(state["exp_avg"], 0.3))

    def test_manifest_rejects_different_world_size(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            rank_files = _kt_optimizer_rank_files(2)
            with open(os.path.join(checkpoint, KT_OPTIMIZER_INDEX_NAME), "w", encoding="utf-8") as handle:
                json.dump({"world_size": 2, "rank_files": rank_files}, handle)

            with self.assertRaisesRegex(RuntimeError, "saved with world_size=2"):
                _read_kt_optimizer_manifest(checkpoint, expected_world_size=4)

    def test_manifest_rejects_missing_rank_file(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            rank_files = _kt_optimizer_rank_files(2)
            torch.save({}, os.path.join(checkpoint, rank_files[0]))
            with open(os.path.join(checkpoint, KT_OPTIMIZER_INDEX_NAME), "w", encoding="utf-8") as handle:
                json.dump({"world_size": 2, "rank_files": rank_files}, handle)

            with self.assertRaisesRegex(RuntimeError, "missing rank files"):
                _read_kt_optimizer_manifest(checkpoint, expected_world_size=2)

    def test_failed_rank_save_invalidates_old_manifest_without_partial_publish(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            old_manifest = os.path.join(checkpoint, KT_OPTIMIZER_INDEX_NAME)
            with open(old_manifest, "w", encoding="utf-8") as handle:
                json.dump({"world_size": 2, "rank_files": _kt_optimizer_rank_files(2)}, handle)

            trainer = _make_trainer(_make_adamw([1, 1], 1.0), rank=0, world_size=2)
            _disable_checkpoint_collectives(trainer)
            with (
                patch("transformers.trainer.torch.save", side_effect=OSError("disk full")),
                self.assertRaisesRegex(RuntimeError, "disk full"),
            ):
                trainer._save_kt_fsdp_optimizer(checkpoint)

            self.assertFalse(os.path.exists(old_manifest))
            self.assertFalse(os.path.exists(os.path.join(checkpoint, _kt_optimizer_rank_files(2)[0])))
            self.assertEqual([name for name in os.listdir(checkpoint) if name.endswith(".tmp")], [])

    def test_trainer_state_failure_does_not_publish_partial_file(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            trainer_state_path = os.path.join(checkpoint, "trainer_state.json")

            def failing_writer(temporary_path):
                with open(temporary_path, "w", encoding="utf-8") as handle:
                    handle.write('{"partial":')
                raise OSError("injected trainer state failure")

            with self.assertRaisesRegex(OSError, "injected trainer state failure"):
                _atomic_path_save(failing_writer, trainer_state_path)

            self.assertFalse(os.path.exists(trainer_state_path))
            self.assertEqual([name for name in os.listdir(checkpoint) if name.endswith(".tmp")], [])

    def test_legacy_multi_rank_optimizer_file_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            torch.save({}, os.path.join(checkpoint, OPTIMIZER_NAME))
            trainer = _make_trainer(_make_adamw([1, 1], 1.0), rank=1, world_size=2)

            with self.assertRaisesRegex(RuntimeError, "legacy multi-rank KT/FSDP checkpoint"):
                trainer._resolve_kt_fsdp_optimizer(checkpoint)


if __name__ == "__main__":
    unittest.main()
