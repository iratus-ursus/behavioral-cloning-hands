# check_sim.py
"""
Visual smoke-check for Isaac Lab environments.

- Runs a short random episode
- Saves frames from the tiled camera as an MP4 / GIF
- Can also be launched without --headless for interactive viewing
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import imageio

# ------------------------------------------------------------------
# AppLauncher must be first
# ------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visual check of Isaac Lab environments")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument(
    "--task",
    type=str,
    default="reach_site",
    choices=["reach_site", "reach_prop", "lift_brick", "lift_large_box", "place_brick"],
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=120)
parser.add_argument("--seed", type=int, default=50)
parser.add_argument("--output", type=str, default="results/check_sim.mp4")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument(
    "--no_camera",
    action="store_true",
    help="Disable tiled camera (only useful for interactive mode)",
)

args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------
# Imports after AppLauncher
# ------------------------------------------------------------------
import gymnasium as gym
import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import (
    FrankaReachEnvCfg,
)
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper
from torchrl.envs import TransformedEnv, InitTracker, StepCounter


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
        "env_id": "Isaac-Reach-Franka-v0",
        "cfg_class": FrankaReachEnvCfg,
        "episode_length_s": 6.0,
    },
    "lift_large_box": {
        "env_id": "Isaac-Reach-Franka-v0",
        "cfg_class": FrankaReachEnvCfg,
        "episode_length_s": 6.0,
    },
    "place_brick": {
        "env_id": "Isaac-Reach-Franka-v0",
        "cfg_class": FrankaReachEnvCfg,
        "episode_length_s": 8.0,
    },
}


def make_env(task_name: str, num_envs: int = 1, use_camera: bool = True) -> TransformedEnv:
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
            width=640,
            height=480,
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
    return env


def save_video(frames: list[np.ndarray], path: str, fps: int = 30) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving {len(frames)} frames → {path}")
    imageio.mimsave(path, frames, fps=fps)
    print(f"[INFO] Video saved: {path}")


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    use_camera = not args_cli.no_camera
    print(f"[INFO] Task        : {args_cli.task}")
    print(f"[INFO] num_envs    : {args_cli.num_envs}")
    print(f"[INFO] use_camera  : {use_camera}")
    print(f"[INFO] num_steps   : {args_cli.num_steps}")

    env = make_env(
        task_name=args_cli.task,
        num_envs=args_cli.num_envs,
        use_camera=use_camera,
    )

    # Print specs
    print("\n=== Action Spec ===")
    print(env.action_spec)
    print("\n=== Observation Spec ===")
    print(env.observation_spec)

    # Collect frames with random actions
    frames = []
    td = env.reset()

    for step in range(args_cli.num_steps):
        action = env.action_spec.rand()
        td = td.set("action", action)
        td = env.step(td)

        if use_camera and "pixels" in td["next"].keys(include_nested=True):
            # Take the first environment's image
            img = td["next", "pixels"][0].cpu().numpy()

            # Handle both (H, W, C) and (C, H, W)
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))

            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)

            frames.append(img)

        if td["next", "done"].any():
            td = env.reset()

    env.close()

    if frames:
        save_video(frames, args_cli.output, fps=args_cli.fps)
    else:
        print("[WARN] No frames captured (camera disabled or missing 'pixels' key)")

    simulation_app.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()