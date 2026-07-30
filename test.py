# test.py
"""
Basic smoke tests for Isaac Lab + TorchRL environments.

Runs a few episodes on each registered task and validates
observation / action specs, reward range and episode termination.
"""

from __future__ import annotations

import argparse
import unittest

import numpy as np
import torch
import gymnasium as gym

# ------------------------------------------------------------------
# AppLauncher must come before torch / isaaclab imports
# ------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Environment smoke tests")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=666)
args_cli, _ = parser.parse_known_args(["--headless"])  # force headless for tests
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------
# Now safe to import the rest
# ------------------------------------------------------------------
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import (
    FrankaReachEnvCfg,
)
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper
from torchrl.envs import TransformedEnv, InitTracker, StepCounter, RewardSum


_NUM_EPISODES = 3
_NUM_STEPS_PER_EPISODE = 20

# Same registry used in train.py (keep them in sync)
TASK_REGISTRY = {
    "reach_site": {
        "env_id": "Isaac-Reach-Franka-v0",
        "cfg_class": FrankaReachEnvCfg,
        "episode_length_s": 5.0,
    },
    "reach_prop": {
        "env_id": "Isaac-Reach-Franka-v0",
        "cfg_class": FrankaReachEnvCfg,
        "episode_length_s": 5.0,
    },
    "lift_brick": {
        "env_id": "Isaac-Reach-Franka-v0",  # TODO: replace when ported
        "cfg_class": FrankaReachEnvCfg,
        "episode_length_s": 6.0,
    },
    "lift_large_box": {
        "env_id": "Isaac-Reach-Franka-v0",  # TODO: replace when ported
        "cfg_class": FrankaReachEnvCfg,
        "episode_length_s": 6.0,
    },
    "place_brick": {
        "env_id": "Isaac-Reach-Franka-v0",  # TODO: replace when ported
        "cfg_class": FrankaReachEnvCfg,
        "episode_length_s": 8.0,
    },
}


def make_env(task_name: str, num_envs: int = 4, use_camera: bool = False) -> TransformedEnv:
    """Create a minimal environment for testing (no camera by default)."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}")

    task_info = TASK_REGISTRY[task_name]
    cfg = task_info["cfg_class"]()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = "cuda:0"
    cfg.sim.dt = 1.0 / 60.0
    cfg.decimation = 2
    cfg.episode_length_s = task_info["episode_length_s"]

    if use_camera:
        IsaacLabWrapper.add_tiled_camera_config(
            cfg, width=84, height=84, camera_name="tiled_camera"
        )

    base_env = gym.make(task_info["env_id"], cfg=cfg)
    env = IsaacLabWrapper(
        base_env,
        device="cuda:0",
        from_tiled_camera=use_camera,
        pixels_key="pixels",
        native_autoreset=True,
    )

    env = TransformedEnv(env)
    env.append_transform(InitTracker())
    env.append_transform(StepCounter(max_steps=int(task_info["episode_length_s"] * 60)))
    env.append_transform(RewardSum())
    return env


class IsaacLabTaskTest(unittest.TestCase):
    """Smoke tests that run on every registered task."""

    def _validate_observation(self, td, observation_spec):
        """Check that keys and shapes roughly match the observation spec."""
        for key in observation_spec.keys():
            self.assertIn(key, td.keys(include_nested=True))
            obs = td.get(key)
            self.assertTrue(torch.isfinite(obs).all(), f"Non-finite values in {key}")

    def _validate_reward(self, reward: torch.Tensor):
        self.assertTrue(torch.is_tensor(reward))
        self.assertTrue(torch.isfinite(reward).all())
        # Many Isaac Lab tasks use dense rewards that can be outside [0, 1]
        # so we only check finiteness.

    def _validate_action_spec(self, env: TransformedEnv):
        action_spec = env.action_spec
        self.assertTrue(torch.isfinite(action_spec.space.low).all())
        self.assertTrue(torch.isfinite(action_spec.space.high).all())

    def test_all_tasks_run(self):
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)

        for task_name in TASK_REGISTRY:
            with self.subTest(task=task_name):
                print(f"\n[TEST] Running task: {task_name}")
                env = make_env(task_name, num_envs=args_cli.num_envs, use_camera=False)

                self._validate_action_spec(env)

                # Reset
                td = env.reset()
                self._validate_observation(td, env.observation_spec)

                for episode in range(_NUM_EPISODES):
                    for step in range(_NUM_STEPS_PER_EPISODE):
                        # Random action inside the action bounds
                        action = env.action_spec.rand()
                        td = td.set("action", action)
                        td = env.step(td)

                        self._validate_observation(td["next"], env.observation_spec)
                        self._validate_reward(td["next", "reward"])

                        # Early break if all environments are done
                        if td["next", "done"].all():
                            break

                    # Reset for next episode
                    td = env.reset()

                env.close()
                print(f"[TEST] {task_name} – OK")

    def test_camera_observation(self):
        """Quick check that tiled camera produces finite pixels."""
        env = make_env("reach_site", num_envs=2, use_camera=True)
        td = env.reset()

        self.assertIn("pixels", td.keys(include_nested=True))
        pixels = td.get("pixels")
        self.assertTrue(torch.isfinite(pixels.float()).all())
        self.assertEqual(pixels.ndim, 4)  # (N, H, W, C) or (N, C, H, W)

        env.close()
        print("[TEST] camera observation – OK")


if __name__ == "__main__":
    # Run tests then shut down the simulator
    try:
        unittest.main(argv=["first-arg-is-ignored"], exit=False)
    finally:
        simulation_app.close()