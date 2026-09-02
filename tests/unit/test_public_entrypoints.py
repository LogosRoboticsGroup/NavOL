import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_script(relative_path: str, module_name: str):
    path = REPOSITORY_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingEntryPointTests(unittest.TestCase):
    def test_canonical_training_command_exposes_paper_configuration(self):
        module = load_script("scripts/train/train_navol.py", "train_navol")
        self.assertEqual(module.CANONICAL_TRAINING.rollout_steps, 128)
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary_directory:
            assets = Path(temporary_directory)
            command = module.build_training_command(
                repository_root=REPOSITORY_ROOT,
                asset_root=assets,
                python_executable="python",
                num_processes=8,
                run_name="navol_canonical",
            )

        expected_arguments = {
            "--distributed",
            "--headless",
            "--rand_camera",
            "--num_envs",
            "32",
            "--max_iterations",
            "1000",
            "agent.num_steps_per_env=128",
            "agent.algorithm.num_learning_epochs=10",
            "agent.algorithm.num_mini_batches=16",
            "agent.algorithm.learning_rate=1e-5",
            "agent.algorithm.lambda_critic=0.1",
            "env.events.reset_pose.params.height_range=[0.25,1.25]",
            "env.events.reset_pose.params.pitch_range=[-30.0,0.0]",
            "env.actions.joint_combined.use_mpc=true",
            "env.actions.joint_combined.action_type=rand",
            "env.actions.joint_combined.action_rand_p=0.8",
            "env.actions.joint_combined.critic_collision_reduction=mean",
        }
        self.assertTrue(expected_arguments.issubset(set(command)))
        self.assertNotIn("env.actions.joint_combined.action_type=command", command)
        self.assertIn(str((assets / "datasets" / "train" / "3d_front_scene_50").resolve()), command)
        self.assertIn(
            "agent.policy.pretrained_model_path="
            + str((assets / "models" / "navdp-cross-modal.ckpt").resolve()),
            command,
        )

    def test_training_command_consumes_asset_contract_constants(self):
        module = load_script("scripts/train/train_navol.py", "train_navol_asset_contract")
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary_directory:
            assets = Path(temporary_directory)
            with (
                patch.object(module, "DEFAULT_TRAIN_DATASET", "train/contract-scenes"),
                patch.object(module, "DEFAULT_BASE_CHECKPOINT", "contract-base.ckpt"),
            ):
                command = module.build_training_command(
                    repository_root=REPOSITORY_ROOT,
                    asset_root=assets,
                    python_executable="python",
                    num_processes=1,
                    run_name="contract_test",
                )

        self.assertIn(str((assets / "datasets" / "train" / "contract-scenes").resolve()), command)
        self.assertIn(
            "agent.policy.pretrained_model_path="
            + str((assets / "models" / "contract-base.ckpt").resolve()),
            command,
        )


class EvaluationEntryPointTests(unittest.TestCase):
    def test_benchmark_commands_cover_every_scene_and_fixed_pair(self):
        module = load_script("scripts/eval/evaluate_benchmark.py", "evaluate_benchmark")
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            assets = root / "assets"
            output = root / "results"
            commands = module.build_evaluation_commands(
                repository_root=REPOSITORY_ROOT,
                asset_root=assets,
                output_root=output,
                python_executable="python",
                splits=("in_domain", "out_domain"),
            )

        self.assertEqual(len(commands), 16)
        for index, command in enumerate(commands):
            split = "in_domain" if index < 8 else "out_domain"
            scene_index = index % 8
            self.assertIn("--num_envs", command)
            self.assertEqual(command[command.index("--num_envs") + 1], "1")
            self.assertEqual(command[command.index("--num_episodes") + 1], "100")
            self.assertEqual(command[command.index("--scene_index") + 1], str(scene_index))
            self.assertIn(str((assets / "datasets" / "benchmarks" / split).resolve()), command)
            self.assertIn(str((assets / "models" / "checkpoints" / "navol-mpc-iter1000.pt").resolve()), command)
            self.assertIn("--use_mpc", command)
            self.assertNotIn("--wo_critic", command)

    def test_benchmark_commands_consume_policy_checkpoint_constant(self):
        module = load_script("scripts/eval/evaluate_benchmark.py", "evaluate_benchmark_asset_contract")
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            with patch.object(module, "DEFAULT_POLICY_CHECKPOINT", "checkpoints/contract-policy.pt"):
                commands = module.build_evaluation_commands(
                    repository_root=REPOSITORY_ROOT,
                    asset_root=root / "assets",
                    output_root=root / "results",
                    python_executable="python",
                    splits=("in_domain",),
                )

        expected = str((root / "assets" / "models" / "checkpoints" / "contract-policy.pt").resolve())
        for command in commands:
            self.assertEqual(command[command.index("--checkpoint_path") + 1], expected)

    def test_benchmark_commands_accept_an_explicit_checkpoint(self):
        module = load_script("scripts/eval/evaluate_benchmark.py", "evaluate_benchmark_checkpoint")
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "custom-model.pt"
            commands = module.build_evaluation_commands(
                repository_root=REPOSITORY_ROOT,
                asset_root=root / "assets",
                output_root=root / "results",
                python_executable="python",
                splits=("in_domain",),
                checkpoint=checkpoint,
            )

        for command in commands:
            self.assertEqual(
                command[command.index("--checkpoint_path") + 1],
                str(checkpoint.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
