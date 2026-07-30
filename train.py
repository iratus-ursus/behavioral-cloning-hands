from typing import Callable, Optional
import argparse

from tf_agents.environments import tf_py_environment, ObservationFilterWrapper

import core.utility.options as options
from trainer.ppo_trainer import PPOTrainer
from tf_agents.environments.dm_control_wrapper import DmControlWrapper
from tf_agents.train.utils import strategy_utils
from tf_agents.environments import suite_dm_control
import numpy as np
from dm_control import composer
from dm_control.manipulation.shared import constants
from config import obs
from core.utility.wrap import MyDmControlWrapper
from tf_agents.environments import wrappers
import tf_agents
from core.agents.robot import Robot

from envs.arenas import CustomArena
from tasks.utils import lift, reach, move_prop
from tasks.reach_arm import reach_site_vision
from tasks.lift_arm import lift_large_box_vision

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_env():
    task = reach_site_vision()
    train_env = composer.Environment(task)
    train_env = DmControlWrapper(train_env)
    train_env = wrappers.TimeLimit(train_env, duration=300)
    return train_env


def main(modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None):
    parser = options.get_train_parser()
    args = options.parse_args(parser, modify_parser=modify_parser)

    task = reach_site_vision()

    train_env = composer.Environment(task)
    eval_env = composer.Environment(task)

    train_env = DmControlWrapper(train_env)
    eval_env = DmControlWrapper(eval_env)

    train_env = wrappers.TimeLimit(train_env, duration=300)
    eval_env = wrappers.TimeLimit(eval_env, duration=400)

    #train_env = tf_agents.environments.ParallelPyEnvironment([create_env] * 2)

    train_env = tf_py_environment.TFPyEnvironment(train_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_env)

    trainer = PPOTrainer(args=args,
                         env=train_env,
                         eval_env=eval_env)

    trained_agent = trainer.train()


if __name__ == "__main__":
    main()
