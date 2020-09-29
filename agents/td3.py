import os

import numpy as np
import tensorflow as tf

from agents.ddpg_backend.target_update_ops import update_target_variables
from agents.ddpg import DDPG, ActorNetwork
from agents.misc.logger import CmdLineLogger

tf.keras.backend.set_floatx('float64')

class CriticNetwork(tf.keras.Model):

    def __init__(self, units_hidden_layers, dim_obs,  dim_outputs, activations=None, seed=94):
        # call the parent constructor
        super(CriticNetwork, self).__init__()

        self.dim_obs = dim_obs

        # check if default activations should be used (relus)
        if not activations:
            activations = ["relu"] * len(units_hidden_layers)

        # setup the initializers
        self.seed = seed
        glorot_init = tf.keras.initializers.GlorotUniform()
        uniform_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

        # define the layers that are going to be used in our critics
        # --- q1
        self.hidden_layers_q1 = [tf.keras.layers.Dense(dim, activation=activation, kernel_initializer=glorot_init,
                                                       bias_initializer=glorot_init)
                                 for dim, activation in zip(units_hidden_layers, activations)]
        self.out_q1 = tf.keras.layers.Dense(dim_outputs, kernel_initializer=uniform_init, bias_initializer=uniform_init)
        # --- q2
        self.hidden_layers_q2 = [tf.keras.layers.Dense(dim, activation=activation, kernel_initializer=glorot_init,
                                                       bias_initializer=glorot_init)
                                 for dim, activation in zip(units_hidden_layers, activations)]
        self.out_q2 = tf.keras.layers.Dense(dim_outputs, kernel_initializer=uniform_init, bias_initializer=uniform_init)

    def call(self, inputs):
        # define the forward pass of q1
        z = inputs
        for layer in self.hidden_layers_q1:
            z = layer(z)
        q1 = self.out_q1(z)

        # define the forward pass of q2
        k = inputs
        for layer in self.hidden_layers_q2:
            k = layer(k)
        q2 = self.out_q2(k)

        return q1, q2


class TD3(DDPG):
    def __init__(self,
                 argv,
                 action_mode,
                 obs_config,
                 task_class,
                 actor_noise_clipping=0.5,
                 actor_update_frequency=2
                 ):
        """
        :param obs_config: configuration of the observation
        """

        # call parent constructor
        super(TD3, self).__init__(argv=argv, action_mode=action_mode, task_class=task_class, obs_config=obs_config)
        self.policy_stddev = self.sigma
        self.actor_noise_clipping = actor_noise_clipping
        self.actor_update_frequency = actor_update_frequency
        self.max_actions_w_gripper = tf.constant(self.max_actions + [1.0], dtype=tf.float64)
        self.training_step = tf.Variable(0,  dtype=tf.int32)

        # use copying instead of "soft" updates
        self.use_target_copying = False

        # --- define the critic and its target ---
        self.critic = CriticNetwork(self.layers_critic, dim_obs=self.dim_observations, dim_outputs=1)   # one Q-value per state needed
        self.target_critic = CriticNetwork(self.layers_critic, dim_obs=self.dim_observations, dim_outputs=1)    # one Q-value per state needed
        # instantiate the models (if we do not instantiate the model, we can not copy their weights)
        self.critic.build((1, self.dim_inputs_critic))
        self.target_critic.build((1, self.dim_inputs_critic))
        # setup the critic's optimizer
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.lr_critic)

        # --- copy weights to targets or load old model weights
        if type(self) == TD3:
            self.init_or_load_weights()

        TC = tf.keras.callbacks.TensorBoard(log_dir=self.root_log_dir)
        TC.set_model(model=self.actor)
        TC.set_model(model=self.critic)

    @tf.function
    def _compute_td_error(self, states, actions, rewards, next_states, dones):
        not_dones = 1.0 - dones
        next_actions = self.target_actor(next_states)
        # add noise for target policy smoothing regularization
        noise = tf.clip_by_value(tf.random.normal(shape=tf.shape(next_actions), stddev=self.policy_stddev, dtype=tf.float64),
                                 -self.actor_noise_clipping, self.actor_noise_clipping)
        batch_size = tf.shape(next_actions)[0]
        clippings = tf.tile([self.max_actions_w_gripper], [batch_size, 1])
        next_actions = tf.clip_by_value(next_actions + noise, clip_value_min=-clippings, clip_value_max=clippings)
        # calculate the target and the error
        input_target_critic = tf.concat([next_states, next_actions], axis=1)
        target_Q1, target_Q2 = self.target_critic(input_target_critic)
        target_Q = tf.minimum(target_Q1, target_Q2)
        target_Q = rewards + (not_dones * self.gamma * target_Q)
        target_Q = tf.stop_gradient(target_Q)
        input_current_q = tf.concat([states, actions], axis=1)
        current_Q1, current_Q2 = self.critic(input_current_q)
        return tf.squeeze(target_Q - current_Q1), tf.squeeze(target_Q - current_Q2)

    @tf.function
    def train_inner(self, states, actions, rewards, next_states, dones, is_weights=1.0):
        # --- Critic training ---
        with tf.GradientTape() as tape:
            td_errors_q1, td_errors_q2 = self._compute_td_error(states, actions, rewards, next_states, dones)
            mean_td_errors = (td_errors_q1 + td_errors_q2) / 2
            td_errors_q1 = td_errors_q1 * is_weights
            td_errors_q2 = td_errors_q2 * is_weights
            critic_loss = tf.reduce_mean(tf.square(td_errors_q1)) + tf.reduce_mean(tf.square(td_errors_q2))

        # calculate the gradients and optimize
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # --- Actor training ---
        self.training_step.assign_add(1)
        with tf.GradientTape() as tape:
            next_action = self.actor(states)
            input_critic = tf.concat([states, next_action], axis=1)
            actor_loss = -tf.reduce_mean(self.critic(input_critic))

        remainder = tf.math.mod(self.training_step, self.actor_update_frequency)
        if tf.equal(remainder, 0):

            # calculate the gradients and optimize
            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.optimizer_actor.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            # update targets
            update_target_variables(self.target_critic.weights, self.critic.weights, self.tau)
            update_target_variables(self.target_actor.weights, self.actor.weights, self.tau)

        return critic_loss, actor_loss, mean_td_errors

