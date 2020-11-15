import numpy as np
import tensorflow as tf

from rlbench.observation_config import ObservationConfig
from rlbench.action_modes import ActionMode
from agents.base import Agent
import agents.misc.utils as utils


class ESAgent(Agent):

    def __init__(self, action_mode: ActionMode, task_class, obs_config: ObservationConfig, agent_config_path):
        # call parent constructor
        super(ESAgent, self).__init__(action_mode, task_class, obs_config, agent_config_path)

        # setup some parameters
        hparams = self.cfg["ESAgent"]["Hyperparameters"]
        self.global_episode = 0
        self.n_descendants = hparams["n_descendants"]
        self.n_descendants_abs = 2*self.n_descendants    # 2* because we use mirrored sampling
        self.lr = hparams["lr"]
        self.sigma = hparams["sigma"]
        self.layers_network = hparams["layers_network"]
        if self.n_descendants_abs > 1 and not self.headless:
            print("Turning headless mode on, since more than one descendant is running.")
            self.headless = True
        if self.save_weights:
            self.save_weights_interval = utils.adjust_save_interval(self.save_weights_interval, self.n_descendants)


class Network(tf.keras.Model):

    def __init__(self, units_hidden_layers, dim_actions, max_action, activations=None, seed=94):
        # call the parent constructor
        super(Network, self).__init__()

        self.dim_actions = dim_actions

        # check if default activations should be used (relus)
        if not activations:
            activations = ["relu"] * len(units_hidden_layers)

        self.seed = seed
        glorot_init = tf.keras.initializers.GlorotUniform(seed=self.seed)
        uniform_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=self.seed)

        # define the layers that are going to be used in our actor
        self.hidden_layers = [
            tf.keras.layers.Dense(dim, activation=activation, kernel_initializer=glorot_init,
                                  bias_initializer=glorot_init)
            for dim, activation in zip(units_hidden_layers, activations)]
        self.out = tf.keras.layers.Dense(dim_actions, kernel_initializer=uniform_init,
                                         bias_initializer=uniform_init)

        # set action scaling
        self.max_actions = tf.constant([max_action], dtype=tf.float64)

    def call(self, inputs):
        # define the forward pass
        z = inputs
        for layer in self.hidden_layers:
            z = layer(z)
        # we need to scale actions
        arm_actions, gripper_action = tf.split(self.out(z), [7, 1], axis=1)
        arm_actions = tf.nn.tanh(arm_actions)  # we want arm actions to be between -1 and 1
        gripper_action = tf.nn.sigmoid(gripper_action)  # we want gripper actions to be between 0 and 1

        # scale arm actions
        batch_size = tf.shape(arm_actions)[0]
        scale = tf.tile(self.max_actions, [batch_size, 1])
        arm_actions = tf.multiply(arm_actions, scale)

        return tf.concat([arm_actions, gripper_action], axis=1)

