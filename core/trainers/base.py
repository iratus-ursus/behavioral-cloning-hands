# trainer/base.py
"""
The Trainer base class for all algorithms.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torchrl.envs import TransformedEnv
from torchrl.record.loggers import get_logger


class Trainer(ABC):
    """
    The general interface of the trainer.
    All algorithms are inherited from this class.
    """

    def __init__(
        self,
        env: TransformedEnv,
        args: Any,
        use_camera: bool = True,
        device: str = "cuda:0",
    ):
        self.env = env
        self.args = args
        self.use_camera = use_camera
        self.device = device

        self.log_dir = getattr(args, "log_dir", "logs")
        self.checkpoint_dir = getattr(args, "checkpoint_dir", "checkpoints")
        self.seed = getattr(args, "seed", 42)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logger = get_logger(
            "tensorboard",
            logger_name=self.log_dir,
            experiment_name=self.__class__.__name__.lower(),
        )

        torch.manual_seed(self.seed)

    # ------------------------------------------------------------------
    # Mandatory methods implemented by the heirs
    # ------------------------------------------------------------------
    @abstractmethod
    def train(self) -> None:
        """The main training cycle."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Common utilities
    # ------------------------------------------------------------------
    def save(self, iteration: int, extra: Optional[dict] = None) -> str:
        """Saves the checkpoint. extra — additional tensors/states."""
        path = os.path.join(self.checkpoint_dir, f"ckpt_{iteration:06d}.pt")
        payload = {
            "iteration": iteration,
            "args": vars(self.args) if hasattr(self.args, "__dict__") else self.args,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[INFO] Checkpoint saved → {path}")
        return path

    def load(self, path: str) -> dict:
        """Loads the checkpoint and returns its contents."""
        ckpt = torch.load(path, map_location=self.device)
        print(f"[INFO] Loaded checkpoint from {path}")
        return ckpt

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.logger.log_scalar(name, value, step=step)

    def close(self) -> None:
        """Closes the environment and the logger (if necessary)."""
        if hasattr(self.env, "close"):
            self.env.close()