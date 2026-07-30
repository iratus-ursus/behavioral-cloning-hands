"""
Checkpoint utilities for TorchRL policies.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import nn


class Loader:
    """
    Save / load actor & value networks (and optional optimizer state).

    Typical usage inside a Trainer:

        self.loader = Loader(save_dir="checkpoints", name="reach_ppo")
        ...
        self.loader.save(
            iteration=100,
            actor=self.actor,
            value=self.value,
            optim=self.optim,
            extra={"reward": mean_reward},
        )
        ...
        ckpt = self.loader.load("checkpoints/reach_ppo/ckpt_000100.pt")
        self.actor.load_state_dict(ckpt["actor"])
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        name: str = "policy",
        save_interval: int = 1,
    ):
        self.save_dir = Path(save_dir) / name
        self.save_interval = save_interval
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Loader] Checkpoint directory: {self.save_dir}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(
        self,
        iteration: Union[int, str],
        actor: Optional[nn.Module] = None,
        value: Optional[nn.Module] = None,
        optim: Optional[torch.optim.Optimizer] = None,
        extra: Optional[Dict[str, Any]] = None,
        is_final: bool = False,
    ) -> str:
        """
        Save a checkpoint.

        Parameters
        ----------
        iteration : int or str
            Training step / epoch. If is_final=True becomes "FINAL".
        actor, value : nn.Module
            Networks to serialize.
        optim : torch.optim.Optimizer, optional
            Optimizer state.
        extra : dict, optional
            Any additional metadata (rewards, hyper-params, …).
        is_final : bool
            If True, forces the filename to contain "FINAL".
        """
        if is_final:
            iteration = "FINAL"

        filename = f"ckpt_{iteration:06d}.pt" if isinstance(iteration, int) else f"ckpt_{iteration}.pt"
        path = self.save_dir / filename

        payload: Dict[str, Any] = {
            "iteration": iteration,
        }
        if actor is not None:
            payload["actor"] = actor.state_dict()
        if value is not None:
            payload["value"] = value.state_dict()
        if optim is not None:
            payload["optim"] = optim.state_dict()
        if extra is not None:
            payload["extra"] = extra

        torch.save(payload, path)
        print(f"[Loader] Saved → {path}")
        return str(path)

    def save_parameters(self, params: Dict[str, Any], filename: str = "parameters.pkl") -> str:
        """Save a plain dictionary of hyper-parameters as pickle."""
        path = self.save_dir / filename
        with open(path, "wb") as f:
            pickle.dump(params, f)
        print(f"[Loader] Parameters saved → {path}")
        return str(path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(
        self,
        path: str,
        actor: Optional[nn.Module] = None,
        value: Optional[nn.Module] = None,
        optim: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda:0",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and optionally restore networks / optimizer.

        Returns the full checkpoint dictionary.
        """
        ckpt = torch.load(path, map_location=device)
        print(f"[Loader] Loaded ← {path}")

        if actor is not None and "actor" in ckpt:
            actor.load_state_dict(ckpt["actor"], strict=strict)
            print("[Loader] Actor weights restored")
        if value is not None and "value" in ckpt:
            value.load_state_dict(ckpt["value"], strict=strict)
            print("[Loader] Value weights restored")
        if optim is not None and "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
            print("[Loader] Optimizer state restored")

        return ckpt

    def load_parameters(self, filename: str = "parameters.pkl") -> Dict[str, Any]:
        path = self.save_dir / filename
        with open(path, "rb") as f:
            params = pickle.load(f)
        print(f"[Loader] Parameters loaded ← {path}")
        return params

    def latest_checkpoint(self) -> Optional[str]:
        """Return path to the most recent .pt file, or None."""
        files = sorted(self.save_dir.glob("ckpt_*.pt"))
        if not files:
            return None
        return str(files[-1])


# ------------------------------------------------------------------
# Utility: value normalisation
# ------------------------------------------------------------------
def normalize(
    val: np.ndarray,
    current_min: np.ndarray,
    current_max: np.ndarray,
    new_min: float | np.ndarray,
    new_max: float | np.ndarray,
    clip: bool = False,
) -> np.ndarray:
    """
    Linearly map values from [current_min, current_max] → [new_min, new_max].

    Parameters
    ----------
    clip : bool
        If True, clamp the result to the new range.
    """
    val = np.asarray(val, dtype=np.float64)
    current_min = np.asarray(current_min, dtype=np.float64)
    current_max = np.asarray(current_max, dtype=np.float64)
    new_min = np.asarray(new_min, dtype=np.float64)
    new_max = np.asarray(new_max, dtype=np.float64)

    scaled = (new_max - new_min) / (current_max - current_min) * (val - current_min) + new_min

    if clip:
        return np.clip(scaled, new_min, new_max)
    return scaled