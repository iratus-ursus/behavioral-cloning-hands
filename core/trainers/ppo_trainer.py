# trainer/ppo_trainer.py
"""
PPO Trainer based on TorchRL + Isaac Lab.
Inherited from the base Trainer.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal
from torchrl.modules.models import ConvNet, MLP
from tensordict.nn import TensorDictModule, TensorDictSequential

from trainer.base import Trainer
from trainer.ppo_agent import make_networks
from utils.utils import Loader


class PPOTrainer(Trainer):
    """
    On-policy PPO trainer.
    """

    def __init__(
        self,
        env,
        args: Any,
        use_camera: bool = True,
        device: str = "cuda:0",
    ):
        super().__init__(env=env, args=args, use_camera=use_camera, device=device)

        # Hyperparameters
        self.max_iterations = getattr(args, "max_iterations", 2000)
        self.frames_per_batch = getattr(args, "frames_per_batch", 32768)
        self.ppo_epochs = getattr(args, "ppo_epochs", 4)
        self.lr = getattr(args, "lr", 3e-4)
        self.clip_epsilon = getattr(args, "clip_epsilon", 0.2)
        self.entropy_coef = getattr(args, "entropy_coef", 0.01)
        self.gamma = getattr(args, "gamma", 0.99)
        self.lmbda = getattr(args, "lmbda", 0.95)
        self.checkpoint_interval = getattr(args, "checkpoint_interval", 50)

        # Building components
        self.actor, self.value = self._build_policy()
        self.collector = self._build_collector()
        self.advantage = self._build_advantage()
        self.loss_module = self._build_loss()
        self.optim = Adam(self.loss_module.parameters(), lr=self.lr)

        # On-policy buffer
        self.rb = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )

        self.loader = Loader(
            save_dir=self.checkpoint_dir,
            name=f"{getattr(args, 'task', 'reach')}_ppo",
        )

    # ------------------------------------------------------------------
    # Building a policy
    # ------------------------------------------------------------------
    def _build_policy(self):
        return make_networks(
                env=self.env,
                use_camera=self.use_camera,
                device=self.device,
            )

    # ------------------------------------------------------------------
    # Collector / Advantage / Loss
    # ------------------------------------------------------------------
    def _build_collector(self):
        return SyncDataCollector(
            create_env_fn=lambda: self.env,
            policy=self.actor,
            frames_per_batch=self.frames_per_batch,
            total_frames=-1,
            device=self.device,
            storing_device="cpu",
            max_frames_per_traj=300,
        )

    def _build_advantage(self):
        return GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=self.value,
            average_gae=True,
        )

    def _build_loss(self):
        return ClipPPOLoss(
            actor=self.actor,
            critic=self.value,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=True,
            entropy_coef=self.entropy_coef,
            critic_coef=1.0,
            loss_critic_type="l2",
        )

    # ------------------------------------------------------------------
    # The main cycle
    # ------------------------------------------------------------------
    def train(self) -> None:
        print(f"[INFO] Starting PPO training for {self.max_iterations} iterations")
        pbar = tqdm(total=self.max_iterations)

        for iteration, data in enumerate(self.collector):
            # 1. GAE
            with torch.no_grad():
                self.advantage(data)

            data = data.reshape(-1)
            self.rb.extend(data)

            # 2. PPO updates
            loss_vals = None
            for _ in range(self.ppo_epochs):
                for batch in self.rb:
                    loss_vals = self.loss_module(batch)
                    loss = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), max_norm=1.0
                    )
                    self.optim.step()
                    self.optim.zero_grad()

            self.rb.empty()

            # 3. Logging
            reward = data["next", "episode_reward"].mean().item()
            length = data["next", "step_count"].float().mean().item()

            self.log_scalar("train/reward", reward, iteration)
            self.log_scalar("train/episode_length", length, iteration)
            if loss_vals is not None:
                self.log_scalar(
                    "train/loss_objective",
                    loss_vals["loss_objective"].item(),
                    iteration,
                )
                self.log_scalar(
                    "train/loss_critic",
                    loss_vals["loss_critic"].item(),
                    iteration,
                )
                self.log_scalar(
                    "train/loss_entropy",
                    loss_vals["loss_entropy"].item(),
                    iteration,
                )

            pbar.set_description(f"rew={reward:.3f} len={length:.1f}")
            pbar.update(1)

            # 4. Checkpoint
            if iteration > 0 and iteration % self.checkpoint_interval == 0:
                self.loader.save(
                    iteration=iteration,
                    actor=self.actor,
                    value=self.value,
                    optim=self.optim,
                    extra={"mean_reward": reward},
                )

            if iteration >= self.max_iterations:
                break

        pbar.close()
        self.collector.shutdown()
        print("[INFO] PPO training finished.")

    def eval(self, num_episodes: int = 10) -> dict:
        """
        """
        return {"mean_reward": 0.0}