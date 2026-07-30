"""
General training entry-point.
Supports multiple algorithms (--algo) and multiple tasks (--task).
"""

from __future__ import annotations

import argparse
from typing import Callable, Optional

# ------------------------------------------------------------------
# AppLauncher MUST be created BEFORE importing torch
# ------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab + TorchRL training")
AppLauncher.add_app_launcher_args(parser)

# Common arguments
parser.add_argument(
    "--algo",
    type=str,
    default="ppo",
    choices=["ppo", "bc"],
    help="Training algorithm",
)
parser.add_argument(
    "--task",
    type=str,
    default="reach_site",
    choices=[
        "reach_site",
        "reach_prop",
        "lift_brick",
        "lift_large_box",
        "place_brick",
    ],
    help="Task to train",
)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument(
    "--no_camera",
    action="store_true",
    help="Use state observations only (disable tiled camera)",
)

# PPO-specific
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--frames_per_batch", type=int, default=32768)
parser.add_argument("--ppo_epochs", type=int, default=4)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--clip_epsilon", type=float, default=0.2)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lmbda", type=float, default=0.95)
parser.add_argument("--checkpoint_interval", type=int, default=50)

# Behavioral Cloning specific
parser.add_argument(
    "--demo_path",
    type=str,
    default=None,
    help="Path to demonstration data (.pt / .hdf5) required for BC",
)
parser.add_argument("--bc_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)

args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------
# Now it is safe to import torch and the rest
# ------------------------------------------------------------------
import os
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import (
    FrankaReachEnvCfg,
)
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper
from torchrl.envs import TransformedEnv, InitTracker, StepCounter, RewardSum


# Mapping from CLI task name → Isaac Lab environment id / config
# (extend this dict when you port more dm_control tasks)
TASK_REGISTRY = {
    "reach_site": {
        "env_id": "Isaac-Reach-Franka-v0",
        "cfg_class": FrankaReachEnvCfg,
        "episode_length_s": 5.0,
    },
    # The following are placeholders – replace with real Isaac Lab
    # configs once the corresponding tasks are ported.
    "reach_prop": {
        "env_id": "Isaac-Reach-Franka-v0",
        "cfg_class": FrankaReachEnvCfg,
        "episode_length_s": 5.0,
    },
    "lift_brick": {
        "env_id": "Isaac-Reach-Franka-v0",          # TODO: replace
        "cfg_class": FrankaReachEnvCfg,
        "episode_length_s": 6.0,
    },
    "lift_large_box": {
        "env_id": "Isaac-Reach-Franka-v0",          # TODO: replace
        "cfg_class": FrankaReachEnvCfg,
        "episode_length_s": 6.0,
    },
    "place_brick": {
        "env_id": "Isaac-Reach-Franka-v0",          # TODO: replace
        "cfg_class": FrankaReachEnvCfg,
        "episode_length_s": 8.0,
    },
}


def make_env(
    task_name: str,
    num_envs: int = 1024,
    use_camera: bool = True,
) -> TransformedEnv:
    """Create and wrap the selected Isaac Lab environment."""
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
            cfg,
            width=84,
            height=84,
            camera_name="tiled_camera",
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


def main(modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None) -> None:
    # Optional extra parser modifications (kept for compatibility)
    if modify_parser is not None:
        modify_parser(parser)
        # re-parse if needed – currently args_cli already exists

    torch.manual_seed(args_cli.seed)
    use_camera = not args_cli.no_camera

    os.makedirs(args_cli.log_dir, exist_ok=True)
    os.makedirs(args_cli.checkpoint_dir, exist_ok=True)

    print(f"[INFO] Algorithm : {args_cli.algo.upper()}")
    print(f"[INFO] Task      : {args_cli.task}")
    print(f"[INFO] num_envs  : {args_cli.num_envs}")
    print(f"[INFO] camera    : {use_camera}")

    env = make_env(
        task_name=args_cli.task,
        num_envs=args_cli.num_envs,
        use_camera=use_camera,
    )

    if args_cli.algo == "ppo":
        from trainer.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            env=env,
            args=args_cli,
            use_camera=use_camera,
        )
        trainer.train()

    elif args_cli.algo == "bc":
        from trainer.bc_trainer import BCTrainer

        if args_cli.demo_path is None:
            raise ValueError("Behavioral Cloning requires --demo_path")

        trainer = BCTrainer(
            env=env,
            demo_path=args_cli.demo_path,
            args=args_cli,
            use_camera=use_camera,
        )
        trainer.train()

    else:
        raise ValueError(f"Unknown algorithm: {args_cli.algo}")

    env.close()
    simulation_app.close()
    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()