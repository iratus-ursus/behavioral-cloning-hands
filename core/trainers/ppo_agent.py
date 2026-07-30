# trainer/ppo_agent.py
"""
Policy / network factories for PPO (TorchRL).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal
from torchrl.modules.models import ConvNet, MLP
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import TransformedEnv


def make_encoder(use_camera: bool, env: TransformedEnv) -> Tuple[TensorDictModule, int, list]:
    """
    Build observation encoder.

    Returns
    -------
    encoder : TensorDictModule
    feature_dim : int
    value_in_keys : list[str]
    """
    if use_camera:
        cnn = ConvNet(
            num_cells=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            activation_class=nn.ReLU,
            aggregator_class=nn.AdaptiveAvgPool2d,
            aggregator_kwargs={"output_size": (1, 1)},
            squeeze_output=True,
        )
        encoder = TensorDictModule(
            cnn,
            in_keys=["pixels"],
            out_keys=["features"],
        )
        feature_dim = 64
        value_in_keys = ["pixels"]
    else:
        obs_dim = env.observation_spec["policy"].shape[-1]
        encoder = TensorDictModule(
            nn.Identity(),
            in_keys=["policy"],
            out_keys=["features"],
        )
        feature_dim = obs_dim
        value_in_keys = ["policy"]

    return encoder, feature_dim, value_in_keys


def make_backbone(feature_dim: int, hidden_dim: int = 256) -> TensorDictModule:
    """Shared MLP backbone after the encoder."""
    return TensorDictModule(
        MLP(
            in_features=feature_dim,
            out_features=hidden_dim,
            num_cells=[512, 512],
            activation_class=nn.ReLU,
        ),
        in_keys=["features"],
        out_keys=["hidden"],
    )


def make_actor(
    env: TransformedEnv,
    encoder: TensorDictModule,
    backbone: TensorDictModule,
    device: str = "cuda:0",
) -> ProbabilisticActor:
    """Create ProbabilisticActor with TanhNormal distribution."""
    action_dim = env.action_spec.shape[-1]

    actor_head = TensorDictModule(
        nn.Linear(256, 2 * action_dim),  # loc + scale
        in_keys=["hidden"],
        out_keys=["loc_scale"],
    )

    actor_module = TensorDictSequential(encoder, backbone, actor_head)

    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["loc_scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
            "tanh_loc": True,
        },
        return_log_prob=True,
    ).to(device)

    return actor


def make_value(
    encoder: TensorDictModule,
    backbone: TensorDictModule,
    value_in_keys: list,
    device: str = "cuda:0",
) -> ValueOperator:
    """Create value network (critic)."""
    value_head = TensorDictModule(
        nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ),
        in_keys=["hidden"],
        out_keys=["state_value"],
    )

    value = ValueOperator(
        module=TensorDictSequential(encoder, backbone, value_head),
        in_keys=value_in_keys,
    ).to(device)

    return value


def make_networks(
    env: TransformedEnv,
    use_camera: bool = True,
    device: str = "cuda:0",
) -> Tuple[ProbabilisticActor, ValueOperator]:
    """
    High-level factory used by PPOTrainer.

    Returns
    -------
    actor : ProbabilisticActor
    value : ValueOperator
    """
    encoder, feature_dim, value_in_keys = make_encoder(use_camera, env)
    backbone = make_backbone(feature_dim)

    actor = make_actor(env, encoder, backbone, device=device)
    value = make_value(encoder, backbone, value_in_keys, device=device)

    return actor, value