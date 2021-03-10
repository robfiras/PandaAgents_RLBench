import sys

import numpy as np
import tensorflow as tf

from rlbench.observation_config import ObservationConfig
from rlbench.action_modes import ActionMode
from agents.base import Agent
import agents.misc.utils as utils


class ESAgent(Agent):

    def __init__(self, action_mode: ActionMode,  obs_config: ObservationConfig, task_class, agent_config):
        # call parent constructor
        super(ESAgent, self).__init__(action_mode, obs_config, task_class, agent_config)

        # setup some parameters
        self.es_hparams = self.cfg["ESAgent"]["Hyperparameters"]
        self.n_workers = self.es_hparams["n_workers"]
        self.perturbations_per_batch = self.es_hparams["perturbations_per_batch"]
        if self.n_workers > 1 and not self.headless:
            print("Turning headless mode on, since more than one worker is running.")
            self.headless = True
        if self.perturbations_per_batch % self.n_workers != 0:
            corrected_perturbations_per_batch = self.perturbations_per_batch +\
                                           (self.n_workers - self.perturbations_per_batch % self.n_workers)
            print("\nChanging the number of peturbations per batch from %d to %d." % (self.perturbations_per_batch,
                                                                                      corrected_perturbations_per_batch))
            self.perturbations_per_batch = corrected_perturbations_per_batch

        # correct validation interval
        if self.make_validation_during_training:
            # change number of validation episodes to match number of workers
            if self.validation_interval >= self.perturbations_per_batch:
                remainder = self.validation_interval % self.perturbations_per_batch
            else:
                remainder = self.perturbations_per_batch % self.validation_interval
            if remainder != 0:
                if self.validation_interval >= self.perturbations_per_batch:
                    new_valid_interval = self.validation_interval + (self.n_workers - remainder)
                else:
                    new_valid_interval = self.validation_interval + remainder
                if new_valid_interval - self.validation_interval > 20:
                    question = "Validation interval need to be adjusted from %d to %d. The difference is quite huge, " \
                               "do you want to proceed anyway?" % (self.validation_interval, new_valid_interval)
                    if not utils.query_yes_no(question):
                        print("Terminating ...")
                        sys.exit()
                print("\nChanging validation interval from %d to %d to align with number of workers.\n" %
                      (self.validation_interval, new_valid_interval))
                self.validation_interval = new_valid_interval

        if self.save_weights:
            self.save_weights_interval = utils.adjust_save_interval(self.save_weights_interval, self.n_workers)


class Network(tf.keras.Model):

    def __init__(self, units_hidden_layers, dim_actions, max_action, activations=None, seed=94):
        # call the parent constructor
        super(Network, self).__init__()

        self.dim_actions = dim_actions

        # check if default activations should be used (relus)
        if not activations:
            activations = ["tanh"] * len(units_hidden_layers)

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

