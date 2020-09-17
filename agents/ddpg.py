import os
import time
import datetime

import numpy as np
import tensorflow as tf
from rlbench.backend.observation import Observation

from agents.ddpg_backend.target_update_ops import update_target_variables
from agents.ddpg_backend.replay_buffer import ReplayBuffer
from agents.ddpg_backend.ou_noise import OUNoise
from agents.misc.logger import CmdLineLogger
from agents.base import Agent

#tf.config.run_functions_eagerly(True)
tf.keras.backend.set_floatx('float64')


class ActorNetwork(tf.keras.Model):

    def __init__(self, units_hidden_layers, dim_actions, max_action, activations=None,
                 sigma=0.1, mu=0.0, use_ou_noise=False, seed=94):
        # call the parent constructor
        super(ActorNetwork, self).__init__()

        self.dim_actions = dim_actions

        # check if default activations should be used (relus)
        if not activations:
            activations = ["relu"] * len(units_hidden_layers)

        self.seed = seed
        glorot_init = tf.keras.initializers.GlorotUniform(seed=self.seed)
        uniform_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

        # define the layers that are going to be used in our actor
        self.hidden_layers = [tf.keras.layers.Dense(dim, activation=activation, kernel_initializer=glorot_init,
                                                    bias_initializer=glorot_init)
                              for dim, activation in zip(units_hidden_layers, activations)]
        self.out = tf.keras.layers.Dense(dim_actions, kernel_initializer=uniform_init, bias_initializer=uniform_init)

        # set action scaling
        self.max_actions = tf.constant([max_action], dtype=tf.float64)

        # setup the noise for the actor
        self.sigma = sigma
        self.mu = mu
        # check if ou noise should be used
        self.use_ou_noise = use_ou_noise
        if self.use_ou_noise:
            self.ou_noise = OUNoise(dim_actions, mu=self.mu, sigma=self.sigma, seed=self.seed)

    def call(self, inputs):
        # define the forward pass
        z = inputs
        for layer in self.hidden_layers:
            z = layer(z)
        # we need to scale actions
        arm_actions, gripper_action = tf.split(self.out(z), [7, 1], axis=1)
        arm_actions = tf.nn.tanh(arm_actions)           # we want arm actions to be between -1 and 1
        gripper_action = tf.nn.sigmoid(gripper_action)  # we want gripper actions to be between 0 and 1

        # scale arm actions
        batch_size = tf.shape(arm_actions)[0]
        scale = tf.tile(self.max_actions, [batch_size, 1])
        arm_actions = tf.multiply(arm_actions, scale)

        return tf.concat([arm_actions, gripper_action], axis=1)

    def noisy_predict(self, obs, return_tensor=False):
        """
        Makes predictions and adds noise to it
        :param obs: observation
        :param return_tensor: if true, tensor is returned else numpy array
        :return: either numpy array or tensor
        """
        # get prediction without noise
        pred = self.predict(obs)

        # add noise
        if self.use_ou_noise:
            pred = self.add_ou_noise(pred)
        else:
            pred = self.add_gaussian_noise(pred)

        return pred if return_tensor else pred.numpy()

    def add_gaussian_noise(self, predictions):
        """ adds noise from a normal (Gaussian) distribution """
        noisy_predictions = predictions + tf.random.normal(shape=predictions.shape, mean=self.mu, stddev=self.sigma,
                                                           dtype=tf.float32, seed=self.seed)
        return noisy_predictions

    def add_ou_noise(self, predictions):
        """ adds noise from a Ornstein-Uhlenbeck process """
        noisy_predictions = predictions + self.ou_noise.noise(predictions)
        return noisy_predictions


class CriticNetwork(tf.keras.Model):

    def __init__(self, units_hidden_layers, dim_obs,  dim_outputs, activations=None, seed=94):
        # call the parent constructor
        super(CriticNetwork, self).__init__()

        self.dim_obs = dim_obs

        # check if default activations should be used (relus)
        if not activations:
            activations = ["relu"] * len(units_hidden_layers)

        self.seed = seed
        glorot_init = tf.keras.initializers.GlorotUniform(seed=self.seed)
        uniform_init = tf.keras.initializers.RandomUniform(minval=-0.0003, maxval=0.0003)

        # define the layers that are going to be used in our critic
        self.hidden_layers = [tf.keras.layers.Dense(dim, activation=activation, kernel_initializer=glorot_init,
                                                    bias_initializer=glorot_init)
                              for dim, activation in zip(units_hidden_layers, activations)]
        self.out = tf.keras.layers.Dense(dim_outputs, kernel_initializer=uniform_init, bias_initializer=uniform_init)

    def call(self, inputs):
        # define the forward pass
        z = inputs
        for layer in self.hidden_layers:
            z = layer(z)
        return self.out(z)


class DDPG(Agent):
    def __init__(self,
                 argv,
                 action_mode,
                 obs_config,
                 task_class,
                 gamma=0.99,
                 tau=0.001,
                 sigma=0.4,
                 batch_size=64,
                 episode_length=40,
                 training_interval=1,
                 start_training=500000,
                 save_weights_interval=400,
                 use_ou_noise=False,
                 buffer_size=500000,
                 lr_actor=0.0001,
                 lr_critic=0.001,
                 layers_actor=[400, 300],
                 layers_critic=[400, 300],
                 seed=94
                 ):
        """
        :param obs_config: configuration of the observation
        :param gamma: discount factor
        :param tau: hyperparameter soft target updates
        :param sigma: standard deviation for noise
        :param batch_size: batch size to be sampled from the buffer
        :param episode_length: length of an episode
        :param training_interval: intervals used for training (default==1 -> train ech step)
        :param start_training: step at which training is started
        :param save_weights_interval: interval in which the weights are saved
        :param use_ou_noise: if true, Ornstein-Uhlenbeck noise is used instead of Gaussian noise
        :param buffer_size: size of the replay buffer
        :param lr_actor: learning rate actor
        :param lr_critic: learning rate critic
        :param layers_actor: number of units in each dense layer in the actor (len of list defines number of layers)
        :param layers_critic: number of units in each dense layer in the actor (len of list defines number of layers)
        """

        # call parent constructor
        super(DDPG, self).__init__(action_mode, task_class, obs_config, argv, seed)

        # define the dimensions
        self.dim_inputs_actor = self.dim_observations
        self.dim_inputs_critic = self.dim_observations + self.dim_actions

        # setup the some hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.training_interval = training_interval
        self.start_training = start_training
        self.global_step_main = 0
        self.global_episode = 0
        self.use_ou_noise = use_ou_noise

        if not self.training_episodes:
            self.training_episodes = 1000   # default value
        if self.save_weights:
            self.save_weights_interval = save_weights_interval
        # add an custom/unique id for logging
        if (self.use_tensorboard or self.save_weights) and self.root_log_dir:
            if not self.run_id:
                self.run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
            self.root_log_dir = os.path.join(self.root_log_dir, self.run_id, "")
        # setup tensorboard
        self.summary_writer = None
        if self.use_tensorboard:
            self.summary_writer = tf.summary.create_file_writer(logdir=self.root_log_dir)

        # setup the replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size,
                                          path_to_db_write=self.root_log_dir,
                                          path_to_db_read=self.path_to_read_buffer,
                                          dim_observations=self.dim_observations,
                                          dim_actions=self.dim_actions,
                                          write=self.write_buffer)

        # --- define actor and its target---
        self.max_actions = self.task.get_joint_upper_velocity_limits()
        self.actor = ActorNetwork(layers_actor, self.dim_actions, self.max_actions, sigma=sigma, use_ou_noise=use_ou_noise)
        self.target_actor = ActorNetwork(layers_actor,  self.dim_actions, self.max_actions, sigma=sigma, use_ou_noise=use_ou_noise)
        # instantiate the models (if we do not instantiate the model, we can not copy their weights)
        self.actor.build((1, self.dim_inputs_actor))
        self.target_actor.build((1, self.dim_inputs_actor))
        # check if we need to load weights
        if self.path_to_model:
            print("Loading weights from %s to actor..." % self.path_to_model)
            self.actor.load_weights(os.path.join(self.path_to_model, "actor", "variables", "variables"))
            self.target_actor.load_weights(os.path.join(self.path_to_model, "actor", "variables", "variables"))
        else:
            # copy the weights to the target actor
            update_target_variables(self.target_actor.weights, self.actor.weights, tau=1.0)
        # setup the actor's optimizer
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)

        # --- define the critic and its target ---
        self.critic = CriticNetwork(layers_critic, dim_obs=self.dim_observations, dim_outputs=1)   # one Q-value per state needed
        self.target_critic = CriticNetwork(layers_critic, dim_obs=self.dim_observations, dim_outputs=1)    # one Q-value per state needed
        # instantiate the models (if we do not instantiate the model, we can not copy their weights)
        self.critic.build((1, self.dim_inputs_critic))
        self.target_critic.build((1, self.dim_inputs_critic))
        # check if we need to load weights
        if self.path_to_model:
            print("Loading weights from %s to critic..." % self.path_to_model)
            self.critic.load_weights(os.path.join(self.path_to_model, "critic", "variables", "variables"))
            self.target_critic.load_weights(os.path.join(self.path_to_model, "critic", "variables", "variables"))
        else:
            # copy the weights to the target critic
            update_target_variables(self.target_critic.weights, self.critic.weights, tau=1.0)
        # setup the critic's optimizer
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)

    def run(self):
        obs = []
        action = []
        reward = []
        next_obs = []
        done = []
        running_workers = []
        number_of_succ_episodes = 0
        logger = CmdLineLogger(10, self.training_episodes, self.n_workers)
        while self.global_episode < self.training_episodes:
            # reset episode if maximal length is reached or all worker are done
            if self.global_step_main % self.episode_length == 0 or not running_workers:
                obs = []
                # init a list of worker (connections) which haven't finished their episodes yet -> all at reset
                running_workers = self.worker_conn.copy()
                # reset workers
                self.all_worker_reset(running_workers, obs)

                if self.global_step_main != 0:
                    self.global_episode += self.n_workers

                logger(self.global_episode, number_of_succ_episodes)

            # predict action with actor
            action = self.get_action(obs, noise=(False if self.no_training else True))

            # make a step in workers
            self.all_worker_step(obs=obs, reward=reward, action=action, next_obs=next_obs,
                                 done=done, running_workers=running_workers)

            # train if conditions are met
            total_steps = self.global_step_main * (1 + self.n_workers)
            cond_train = (total_steps >= self.start_training and
                          self.global_step_main % self.training_interval == 0 and
                          not self.no_training)

            if cond_train:
                avg_crit_loss = 0
                avg_act_loss = 0
                for i in range(self.n_workers):
                    crit_loss, act_loss = self.train()
                    avg_crit_loss += crit_loss
                    avg_act_loss += act_loss
                avg_crit_loss_loss = avg_crit_loss / self.n_workers
                avg_act_loss = avg_act_loss / self.n_workers

            # save weights if needed
            if self.save_weights and total_steps % self.save_weights_interval == 0 and total_steps > self.start_training:
                self.save_all_models()

            # log to tensorboard if needed
            if self.use_tensorboard and cond_train:
                with self.summary_writer.as_default():
                    # pass logging data to tensorboard
                    tf.summary.scalar('Critic-Loss', avg_crit_loss_loss, step=total_steps)
                    tf.summary.scalar('Actor-Loss', avg_act_loss, step=total_steps)
                    scalar_reward = 0
                    for r in reward:
                        scalar_reward += r
                    tf.summary.scalar('Reward', float(scalar_reward), step=total_steps)
                    tf.summary.scalar('Number of successful Episodes', float(number_of_succ_episodes), step=total_steps)

            for r in reward:
                if r > 0.0:
                    number_of_succ_episodes += 1

            # increment and save next_obs as obs
            self.global_step_main += 1
            obs = next_obs
            next_obs = []
            reward = []
            done = []

        self.clean_up()
        print('\nDone.\n')

    def run_training_only(self):
        if not self.path_to_read_buffer:
            raise AttributeError("Can not run training only if no buffer is provided! Please provide buffer.")

        while self.global_episode < self.training_episodes:
            crit_loss, act_loss = self.train()

            # save weights if needed
            if self.save_weights and self.global_step_main % self.save_weights_interval == 0:
                self.save_all_models()

            # log to tensorboard if needed
            if self.use_tensorboard:
                with self.summary_writer.as_default():
                    # determine the total number of steps made in all threads
                    total_steps = self.global_step_main * (1 + self.n_workers)
                    # pass logging data to tensorboard
                    tf.summary.scalar('Critic-Loss', crit_loss, step=total_steps)
                    tf.summary.scalar('Actor-Loss', act_loss, step=total_steps)

            self.global_step_main += 1

        self.clean_up()
        print('\nDone.\n')


    def all_worker_reset(self, running_workers, obs):
        # queue command to worker
        for w in running_workers:
            w["command_queue"].put(("reset", ()))
        # collect data from workers
        for w in running_workers:
            descriptions, single_obs = w["result_queue"].get()
            obs.append(single_obs)

    def all_worker_step(self, obs, action, reward, next_obs, done, running_workers):
        for w, a in zip(running_workers, action):
            w["command_queue"].put(("step", (a,)))

        # collect results from workers
        finished_workers = []
        for w, o, a, e in zip(running_workers, obs, action, range(self.n_workers)):
            single_next_obs, single_reward, single_done = w["result_queue"].get()
            # single_reward = self.cal_custom_reward(single_next_obs)  # added custom reward
            single_reward = single_reward*10
            single_next_obs = single_next_obs.get_low_dim_data()
            self.replay_buffer.append(o, a, float(single_reward), single_next_obs,
                                      float(single_done), (e+self.global_episode))
            next_obs.append(single_next_obs)
            reward.append(single_reward)
            done.append(single_done)
            if single_done:
                # remove worker from running_workers if finished
                finished_workers.append(w)

        for w in finished_workers:
            running_workers.remove(w)

    def get_action(self, obs, noise=True):
        """
        Predicts an action using the actor network. DO NOT USE WHILE TRAINING. Use predict() or noisy_predict() instead
        :param obs: received observation
        :param noise: if true adds noise to actions tensor
        :return: returns an numpy array containing the corresponding actions
        """
        if noise:
            actions = self.actor.noisy_predict(tf.constant(obs))
        else:
            actions = self.actor.predict(tf.constant(obs))

        return actions

    def cal_custom_reward(self, obs: Observation):
        max_precision = 0.01    # 1cm
        max_reward = 1/max_precision
        scale = 0.1
        gripper_pos = obs.gripper_pose[0:3]         # gripper x,y,z
        target_pos = obs.task_low_dim_state         # target x,y,z
        dist = np.sqrt(np.sum(np.square(np.subtract(target_pos, gripper_pos)), axis=0))     # euclidean norm
        reward = min((1/(dist + 0.00001)), max_reward)
        reward = scale * reward
        return reward

    def save_all_models(self):
        path_to_dir = os.path.join(self.root_log_dir, "weights", "")
        self.actor.save(path_to_dir + "actor")
        # self.target_actor.save(path_to_dir + "target_actor")
        self.critic.save(path_to_dir + "critic")
        # self.target_critic.save(path_to_dir + "target_critic")

    @tf.function
    def _compute_td_error(self, states, actions, rewards, next_states, dones):
        not_dones = 1.0 - dones
        next_actions = self.target_actor(next_states)
        input_target_critic = tf.concat([next_states, next_actions], axis=1)
        target_Q = self.target_critic(input_target_critic)
        target_Q = rewards + (not_dones * self.gamma * target_Q)
        target_Q = tf.stop_gradient(target_Q)
        input_current_q = tf.concat([states, actions], axis=1)
        current_Q = self.critic(input_current_q)
        td_errors = target_Q - current_Q
        return tf.squeeze(td_errors)

    def train(self):
        # sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)

        states = tf.constant(states)
        actions = tf.constant(actions)
        rewards = tf.constant(rewards)
        next_states = tf.constant(next_states)
        dones = tf.constant(dones)

        # call the inner function
        crit_loss, act_loss = self.train_inner(states, actions, rewards, next_states, dones)
        return crit_loss, act_loss

    @tf.function
    def train_inner(self, states, actions, rewards, next_states, dones):
        # --- Critic training ---
        with tf.GradientTape() as tape:
            td_errors = self._compute_td_error(states, actions, rewards, next_states, dones)

            # lets use a squared error for errors less than 1 and a linear error for errors greater than 1 to reduce
            # the impact of very large errors
            td_errors = tf.abs(td_errors)
            clipped_td_errors = tf.clip_by_value(td_errors, 0.0, 1.0)
            linear_errors = 2 * (td_errors - clipped_td_errors)
            critic_loss = tf.reduce_mean(tf.square(clipped_td_errors) + linear_errors)

        # calculate the gradients and optimize
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # --- Actor training ---
        with tf.GradientTape() as tape:
            next_action = self.actor(states)
            input_critic = tf.concat([states, next_action], axis=1)
            actor_loss = -tf.reduce_mean(self.critic(input_critic))

        # calculate the gradients and optimize
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        # update target weights
        update_target_variables(self.target_critic.weights, self.critic.weights, self.tau)
        update_target_variables(self.target_actor.weights, self.actor.weights, self.tau)

        return critic_loss, actor_loss

    def clean_up(self):
        print("Cleaning up ...")
        # shut down all environments
        [q.put(("kill", ())) for q in self.command_queue]
        [worker.join() for worker in self.workers]
        self.env.shutdown()
        # delete replay buffer (this safes the replay buffer one last time)
        del self.replay_buffer
