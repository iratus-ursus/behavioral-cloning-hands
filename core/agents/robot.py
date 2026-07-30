# robot.py
"""
Robot definition utilities for Isaac Lab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import (
    ContactSensorCfg,
    CameraCfg,
    TiledCameraCfg,
)
from isaaclab.utils import configclass


# ------------------------------------------------------------------
# Default offsets / constants (kept for compatibility with old code)
# ------------------------------------------------------------------
ROBOT_OFFSET = (0.0, 0.0, 0.12)


# ------------------------------------------------------------------
# Observation groups that used to live in AgentObservables
# ------------------------------------------------------------------
@configclass
class RobotObservationCfg:
    """
    High-level description of which proprioceptive / tactile / visual
    signals we want from the robot.

    These flags are later turned into Observation terms in the
    environment configuration.
    """

    # Proprioception
    joint_pos: bool = True
    joint_vel: bool = True
    joint_torque: bool = True
    actuator_activation: bool = False          # "act" in MuJoCo

    # Tactile
    fingertip_contact: bool = True
    fingerpad_contact: bool = False

    # IMU-like
    gyro: bool = False
    accelerometer: bool = False

    # Vision
    egocentric_camera: bool = False
    egocentric_width: int = 84
    egocentric_height: int = 84


# ------------------------------------------------------------------
# Helper that builds the common observation terms
# ------------------------------------------------------------------
def build_robot_observation_terms(
    robot_name: str = "robot",
    cfg: Optional[RobotObservationCfg] = None,
) -> Dict[str, dict]:
    """
    Returns a dictionary that can be plugged into an Isaac Lab
    ObservationsCfg.

    Example usage inside an EnvCfg:

        from robots.robot import build_robot_observation_terms, RobotObservationCfg
        obs_cfg = build_robot_observation_terms("robot", RobotObservationCfg())
        self.observations.policy = ObservationsCfg.ObsGroupCfg(
            terms=obs_cfg,
            ...
        )
    """
    if cfg is None:
        cfg = RobotObservationCfg()

    terms: Dict[str, dict] = {}

    if cfg.joint_pos:
        terms["joint_pos"] = {
            "func": "isaaclab.envs.mdp.joint_pos_rel",
            "params": {"asset_cfg": {"name": robot_name}},
        }
    if cfg.joint_vel:
        terms["joint_vel"] = {
            "func": "isaaclab.envs.mdp.joint_vel_rel",
            "params": {"asset_cfg": {"name": robot_name}},
        }
    if cfg.joint_torque:
        # approximated via applied torque / effort
        terms["joint_effort"] = {
            "func": "isaaclab.envs.mdp.joint_effort",
            "params": {"asset_cfg": {"name": robot_name}},
        }

    if cfg.fingertip_contact:
        terms["fingertip_forces"] = {
            "func": "isaaclab.envs.mdp.body_incoming_wrench",
            "params": {
                "asset_cfg": {"name": robot_name},
                # body names must match the USD
                "body_names": [".*fingertip.*"],
            },
        }

    if cfg.egocentric_camera:
        # Camera itself is added as a sensor; here we only declare
        # that the observation group should include the image.
        terms["egocentric_rgb"] = {
            "func": "isaaclab.envs.mdp.image",
            "params": {
                "sensor_cfg": {"name": "egocentric_camera"},
                "data_type": "rgb",
            },
        }

    return terms


# ------------------------------------------------------------------
# Sensor factory helpers
# ------------------------------------------------------------------
def make_egocentric_camera_cfg(
    width: int = 84,
    height: int = 84,
    prim_path: str = "{ENV_REGEX_NS}/Robot/egocentric_camera",
) -> TiledCameraCfg:
    """Tiled camera attached to the robot (replaces MJCFCamera)."""
    return TiledCameraCfg(
        prim_path=prim_path,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.05),
            rot=(0.5, -0.5, 0.5, -0.5),   # looking forward
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=None,                       # camera already in USD or spawned elsewhere
        width=width,
        height=height,
    )


def make_fingertip_contact_cfg(
    body_names: Sequence[str] = (".*fingertip.*",),
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> ContactSensorCfg:
    """Contact sensors on fingertips (replaces touch sensors)."""
    return ContactSensorCfg(
        prim_path=prim_path,
        history_length=1,
        track_air_time=False,
        force_threshold=1.0,
        filter_prim_paths_expr=list(body_names),
    )


# ------------------------------------------------------------------
# Convenience wrappers around an Articulation
# ------------------------------------------------------------------
class RobotHelper:
    """
    Thin helper that provides the same convenience methods that the
    old dm_control Robot class exposed (randomize joints, RSI, …).
    Works with an already-created isaaclab Articulation.
    """

    def __init__(self, articulation, device: str = "cuda:0"):
        self.art = articulation
        self.device = device

    @property
    def num_joints(self) -> int:
        return self.art.num_joints

    def get_joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (lower, upper) joint position limits."""
        limits = self.art.data.soft_joint_pos_limits  # (N, J, 2)
        return limits[..., 0], limits[..., 1]

    def randomize_arm_joints(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """
        Uniformly sample joint positions inside limits
        (equivalent of old randomize_arm_joints).
        """
        lower, upper = self.get_joint_limits()
        if env_ids is None:
            n = lower.shape[0]
            env_ids = torch.arange(n, device=self.device)

        q = torch.empty_like(lower[env_ids])
        q.uniform_(0.0, 1.0)
        q = lower[env_ids] + q * (upper[env_ids] - lower[env_ids])
        self.art.write_joint_position_to_sim(q, env_ids=env_ids)

    def rsi(self, close_factors: float | Sequence[float] = 0.0,
            env_ids: Optional[torch.Tensor] = None) -> None:
        """
        Reset-to-specific-pose (old rsi method).
        close_factors ∈ [0, 1] interpolates between lower and upper limits.
        """
        lower, upper = self.get_joint_limits()
        if env_ids is None:
            n = lower.shape[0]
            env_ids = torch.arange(n, device=self.device)

        if isinstance(close_factors, (int, float)):
            factors = torch.full(
                (len(env_ids), self.num_joints),
                float(close_factors),
                device=self.device,
            )
        else:
            factors = torch.as_tensor(close_factors, device=self.device)
            factors = factors.expand(len(env_ids), -1)

        q = lower[env_ids] + factors * (upper[env_ids] - lower[env_ids])
        self.art.write_joint_position_to_sim(q, env_ids=env_ids)
        # zero velocity & effort
        self.art.write_joint_velocity_to_sim(
            torch.zeros_like(q), env_ids=env_ids
        )
        self.art.set_joint_effort_target(
            torch.zeros_like(q), env_ids=env_ids
        )