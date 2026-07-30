import tempfile

import tensorflow as tf

from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.train import actor
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.agents import PPOAgent
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.agents import PPOClipAgent


class ImageLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ImageLayer, self).__init__()
        self.reshape = tf.keras.layers.Reshape((84, 84, 3))
        self.rescale = tf.keras.layers.Rescaling(scale=1/255.0)
        self.cv_model = tf.keras.applications.ResNet50(include_top=False,
                                                       weights="imagenet")
        self.cv_model.trainable = False
        self.linear = tf.keras.layers.Dense(64)

    def call(self, x):
        x = self.reshape(x)
        x = self.rescale(x)
        x = self.cv_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.linear(x)
        return x


def make_networks(env,
                  actor_net_layer,
                  value_net_layer,
                  dropout_layer_params):
    obs_spec, act_spec, ts_spec = spec_utils.get_tensor_specs(env)

    actor_net = ActorDistributionNetwork(obs_spec,
                                        act_spec,
                                        fc_layer_params=actor_net_layer,
    )
    value_net = ValueNetwork(obs_spec,
                                fc_layer_params=value_net_layer,
    )
    return actor_net, value_net

def make_agent(env,
               strategy,
               actor_net,
               critic_net,
               lr=4e-4):
    # Now create the agent using the actor and critic networks
    obs_spec, act_spec, ts_spec = spec_utils.get_tensor_specs(env)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    with strategy.scope():
        global_step = tf.compat.v1.train.get_or_create_global_step()

        agent = PPOClipAgent(ts_spec,
                            act_spec,
                            optimizer=optimizer,
                            actor_net=actor_net,
                            value_net=critic_net,
                            train_step_counter=global_step,
                            entropy_regularization=1e-2,
                            importance_ratio_clipping=0.2,
        )
    return agent
