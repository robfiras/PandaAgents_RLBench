import os
import sys

import numpy as np
import tensorflow as tf

from agents.ddpg_backend.target_update_ops import update_target_variables
from agents.ddpg_backend.replay_buffer import ReplayBuffer
from agents.ddpg_backend.prio_replay_buffer import PrioReplayBuffer
from agents.ddpg_backend.ou_noise import OUNoise
from agents.misc.logger import CmdLineLogger
from agents.misc.tensorboard_logger import TensorBoardLogger
from agents.base_rl import RLAgent
import agents.misc.utils as utils

# tf.config.run_functions_eagerly(True)
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
        uniform_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=self.seed)

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

        # clip actions
        max_actions = np.concatenate((self.max_actions.numpy(), [[1.0]]), axis=1)   # append gripper action
        min_actions = np.concatenate((-self.max_actions.numpy(), [[0.0]]), axis=1)   # append gripper action
        pred = tf.clip_by_value(pred, clip_value_min=min_actions, clip_value_max=max_actions)
        return pred if return_tensor else pred.numpy()

    def random_predict(self, obs, return_tensor=False):
        """ makes a random prediction of actions """
        arm_action_limits = np.tile(self.max_actions.numpy(), (obs.shape[0], 1))
        rand_arm_actions = (np.random.rand(obs.shape[0], self.dim_actions-1)-0.5) * (2*arm_action_limits)
        rand_gripper_action = np.random.rand(obs.shape[0], 1)
        pred = np.concatenate((rand_arm_actions, rand_gripper_action), axis=1)
        return tf.constant(pred) if return_tensor else pred

    def add_gaussian_noise(self, predictions):
        """ adds noise from a normal (Gaussian) distribution """
        noisy_predictions = predictions + tf.random.normal(shape=predictions.shape, mean=self.mu, stddev=self.sigma,
                                                           dtype=tf.float64)
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
        uniform_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=self.seed)

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


class DDPG(RLAgent):
    def __init__(self,
                 action_mode,
                 obs_config,
                 task_class,
                 agent_config):

        # call parent constructor
        super(DDPG, self).__init__(action_mode, obs_config, task_class, agent_config)

        # define the dimensions
        self.dim_inputs_actor = self.dim_observations
        self.dim_inputs_critic = self.dim_observations + self.dim_actions

        # setup the some hyperparameters
        hparams = self.cfg["DDPG"]["Hyperparameters"]
        self.gamma = hparams["gamma"]
        self.tau = hparams["tau"]
        self.sigma = hparams["sigma"]
        self.batch_size = hparams["batch_size"]
        self.training_interval = hparams["training_interval"]
        self.max_epsilon = hparams["max_epsilon"]
        self.min_epsilon = hparams["min_epsilon"]
        self.epsilon = self.max_epsilon
        self.epsilon_decay_episodes = hparams["epsilon_decay_episodes"]
        self.layers_actor = hparams["layers_actor"]
        self.layers_critic = hparams["layers_critic"]
        self.lr_actor = hparams["lr_actor"]
        self.lr_critic = hparams["lr_critic"]

        # some DDQN specific setups
        setup = self.cfg["DDPG"]["Setup"]
        self.start_training = setup["start_training"]
        self.use_ou_noise = setup["use_ou_noise"]
        self.use_target_copying = setup["use_target_copying"]
        self.interval_copy_target = setup["interval_copy_target"]
        self.global_step_main = 0
        self.global_episode = 0
        self.write_buffer = setup["write_buffer"]
        self.path_to_read_buffer = None
        if setup["read_buffer_id"]:
            self.path_to_read_buffer = os.path.join(self.root_log_dir, setup["read_buffer_id"], "")
            if not os.path.exists(self.path_to_read_buffer):
                raise FileNotFoundError("The given path to the read database's directory does not exists: %s" %
                                        self.path_to_read_buffer)

        # setup the replay buffer
        self.replay_buffer_mode = setup["replay_buffer_mode"]
        if self.replay_buffer_mode == "VANILLA":
            self.replay_buffer = ReplayBuffer(setup["buffer_size"],
                                              path_to_db_write=self.root_log_dir,
                                              path_to_db_read=self.path_to_read_buffer,
                                              dim_observations=self.dim_observations,
                                              dim_actions=self.dim_actions,
                                              write=self.write_buffer)
        elif self.replay_buffer_mode == "PER_PYTHON":
            self.replay_buffer = PrioReplayBuffer(setup["buffer_size"],
                                                  path_to_db_write=self.root_log_dir,
                                                  path_to_db_read=self.path_to_read_buffer,
                                                  dim_observations=self.dim_observations,
                                                  dim_actions=self.dim_actions,
                                                  write=self.write_buffer,
                                                  use_cpp=False)
        elif self.replay_buffer_mode == "PER_CPP":
            self.replay_buffer = PrioReplayBuffer(setup["buffer_size"],
                                                  path_to_db_write=self.root_log_dir,
                                                  path_to_db_read=self.path_to_read_buffer,
                                                  dim_observations=self.dim_observations,
                                                  dim_actions=self.dim_actions,
                                                  write=self.write_buffer,
                                                  use_cpp=True)
        else:
            raise ValueError("Unsupported replay buffer type. Please choose either VANILLA, PER_PYTHON or PER_CPP.")

        if self.path_to_read_buffer:
            if self.replay_buffer.length >= self.start_training:
                self.start_training = 0
            else:
                self.start_training = self.start_training - self.replay_buffer.length
        self.n_steps_random_actions = 2 * self.start_training
        if self.mode == "online_training":
            print("\nStarting training in %d steps." % self.start_training)
            print("Making completely random actions for the next %d steps. \n" % self.n_steps_random_actions)

        # setup tensorboard
        self.summary_writer = None
        if self.use_tensorboard:
            self.tensorboard_logger = TensorBoardLogger(root_log_dir=self.root_log_dir)

        # --- define actor and its target---
        self.actor = ActorNetwork(self.layers_actor, self.dim_actions, self.max_actions, sigma=self.sigma, use_ou_noise=self.use_ou_noise)
        self.target_actor = ActorNetwork(self.layers_actor,  self.dim_actions, self.max_actions, sigma=self.sigma, use_ou_noise=self.use_ou_noise)
        # instantiate the models
        self.actor.build((1, self.dim_inputs_actor))
        self.target_actor.build((1, self.dim_inputs_actor))
        # setup the actor's optimizer
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.lr_actor)

        # --- define the critic and its target ---
        if type(self) == DDPG:
            self.critic = CriticNetwork(self.layers_critic, dim_obs=self.dim_observations, dim_outputs=1)   # one Q-value per state needed
            self.target_critic = CriticNetwork(self.layers_critic, dim_obs=self.dim_observations, dim_outputs=1)    # one Q-value per state needed
            # instantiate the models
            self.critic.build((1, self.dim_inputs_critic))
            self.target_critic.build((1, self.dim_inputs_critic))
            # setup the critic's optimizer
            self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.lr_critic)

        # --- copy weights to targets or load old model weights ---
        if type(self) == DDPG:
            self.init_or_load_weights()

    def run(self):
        if self.mode == "online_training":
            self.run_online_training()
        elif self.mode == "offline_training":
            self.run_offline_training()
        elif self.mode == "validation":
            if not self.path_to_model:
                question = "You have not set a path to model. Do you really want to validate a random model?"
                if not utils.query_yes_no(question):
                    print("Terminating ...")
                    sys.exit()
            self.run_validation(self.actor)
        else:
            raise ValueError("%\ns mode is not supported in DDPG!\n")

    def run_online_training(self):
        """ Main run method for incrementing the simulation and training the agent """
        obs = []
        reward = []
        next_obs = []
        done = []
        running_workers = []
        number_of_succ_episodes = 0
        percentage_succ = 0.0
        step_in_episode = 0
        logger = CmdLineLogger(self.logging_interval, self.training_episodes, self.n_workers)
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
                    step_in_episode = 0

                logger(self.global_episode, number_of_succ_episodes, percentage_succ)

            total_steps = self.global_step_main * self.n_workers

            # predict action with actor
            if self.n_steps_random_actions > total_steps:
                action = self.get_action(obs, mode="random")
            else:
                action = self.get_action(obs, mode="eps-greedy-random")

            # make a step in workers
            self.all_worker_step(obs=obs, reward=reward, action=action, next_obs=next_obs,
                                 done=done, running_workers=running_workers)

            if total_steps >= self.start_training and self.global_step_main % self.training_interval == 0:
                avg_crit_loss = 0
                avg_act_loss = 0
                worker_training_actors = 0
                for i in range(self.n_workers):
                    crit_loss, act_loss = self.train()
                    avg_crit_loss += crit_loss
                    if act_loss:
                        avg_act_loss += act_loss
                        worker_training_actors += 1
                avg_crit_loss = avg_crit_loss / self.n_workers
                if worker_training_actors != 0:
                    avg_act_loss = avg_act_loss / worker_training_actors
                else:
                    avg_act_loss = None

            # save weights if needed
            if self.save_weights and total_steps % self.save_weights_interval == 0 and total_steps > self.start_training:
                self.save_all_models()

            # log to tensorboard if needed
            if self.use_tensorboard:
                number_of_succ_episodes, percentage_succ = self.tensorboard_logger(total_steps=total_steps,
                                                                                   episode=self.global_episode,
                                                                                   losses={"Critic-Loss": avg_crit_loss,
                                                                                           "Actor-Loss": avg_act_loss},
                                                                                   dones=done,
                                                                                   step_in_episode=step_in_episode,
                                                                                   rewards=reward,
                                                                                   epsilon=self.epsilon)

            # increment and save next_obs as obs
            self.global_step_main += 1
            step_in_episode += 1
            obs = next_obs
            next_obs = []
            reward = []
            done = []

        self.clean_up()
        print('\nDone.\n')

    def run_offline_training(self):
        """ Runs only the training methods without connecting to the simulation. Reading replay buffer needed! """
        if not self.path_to_read_buffer:
            raise AttributeError("Can not run training only if no buffer is provided! Please provide buffer.")

        while self.global_step_main < self.training_episodes:
            crit_loss, act_loss = self.train()

            # save weights if needed
            if self.save_weights and self.global_step_main % self.save_weights_interval == 0:
                self.save_all_models()

            # log to tensorboard if needed
            if self.use_tensorboard:
                with self.summary_writer.as_default():
                    # determine the total number of steps made in all threads
                    total_steps = self.global_step_main * self.n_workers
                    # pass logging data to tensorboard
                    tf.summary.scalar('Critic-Loss', crit_loss, step=total_steps)
                    if act_loss:
                        tf.summary.scalar('Actor-Loss', act_loss, step=total_steps)

            self.global_step_main += 1
            if act_loss:
                print("Training step %d Actor-Loss %f Critic-Loss %f" % (self.global_step_main, act_loss, crit_loss))

        self.clean_up()
        print('\nDone.\n')

    def all_worker_reset(self, running_workers, obs):
        """
        Resets the episodes of all all workers
        :param running_workers: list of running worker connections
        :param obs: list of observations to be filled with initial observations
        """
        # queue command to worker
        for w in running_workers:
            w["command_queue"].put(("reset", ()))
        # collect data from workers
        for w in running_workers:
            descriptions, single_obs = w["result_queue"].get()
            obs.append(single_obs)

    def all_worker_step(self, obs, action, reward, next_obs, done, running_workers):
        """
        Increments the simulation in all running workers, receives the new data and adds it to the replay buffer.
        :param obs: list of observations to be filled
        :param action: list of actions to be filled
        :param reward: list of rewards to be filled
        :param next_obs: list of next observations to be filled
        :param done: list of dones to be filled
        :param running_workers: list of running worker connections
        """
        for w, a in zip(running_workers, action):
            w["command_queue"].put(("step", (a,)))

        # collect results from workers
        finished_workers = []
        for w, o, a, e in zip(running_workers, obs, action, range(self.n_workers)):
            single_next_obs, single_reward, single_done = w["result_queue"].get()
            # single_reward = self.cal_custom_reward_3(single_next_obs, single_done)  # added custom reward
            single_reward = single_reward
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

    def get_action(self, obs, mode):
        """
        Predicts an action using the actor network. DO NOT USE WHILE TRAINING. Use predict() or noisy_predict() instead
        :param obs: received observation
        :param mode: sets the mode -> either greedy, greedy+noise, random, eps-greedy-noise or eps-greedy-random
        :return: returns a numpy array containing the corresponding actions
        """
        # set epsilon-decay if not set yet
        total_steps = self.global_step_main * self.n_workers
        if not self.epsilon_decay_episodes and total_steps > self.n_steps_random_actions:
            self.epsilon_decay_episodes = self.training_episodes - self.global_episode
        # calculate epsilon if decay started
        if self.epsilon_decay_episodes and total_steps > self.n_steps_random_actions:
            epsilon_gradient = ((self.max_epsilon - self.min_epsilon)/self.epsilon_decay_episodes)
            self.epsilon = self.max_epsilon - epsilon_gradient * (self.global_episode - self.training_episodes + self.epsilon_decay_episodes)
            self.epsilon = min(self.epsilon, self.max_epsilon)
            self.epsilon = max(self.epsilon, self.min_epsilon)

        if mode == "greedy":
            actions = self.actor.predict(tf.constant(obs))
        elif mode == "greedy+noise":
            actions = self.actor.noisy_predict(tf.constant(obs))
        elif mode == "random":
            actions = self.actor.random_predict(np.array(obs))
        elif mode == "eps-greedy-random":
            choice = np.random.choice(2, 1, p=[self.epsilon, (1-self.epsilon)])
            if choice == 0:
                actions = self.actor.random_predict(np.array(obs))
            if choice == 1:
                actions = self.actor.noisy_predict(tf.constant(obs))
        else:
            raise ValueError("%s not allowed as mode! Choose either greedy, greedy+noise, random or eps-greedy-random." % mode)

        return actions

    def cal_custom_reward_1(self, obs, done):
        finished = (done or ((self.global_step_main+1) % self.episode_length == 0))
        if finished:
            max_precision = 0.01    # 1cm
            max_reward = 1/max_precision
            scale = 1.0
            gripper_pos = obs[22:25]         # gripper x,y,z
            target_pos = obs[-3:]        # target x,y,z
            dist = np.sqrt(np.sum(np.square(np.subtract(target_pos, gripper_pos)), axis=0))     # euclidean norm
            reward = min((1/(dist + 0.00001)), max_reward)
            reward = scale * reward
            return reward
        else:
            return 0.0

    def cal_custom_reward_2(self, done):
        if done:
            return 10.0
        else:
            return -0.005

    def cal_custom_reward_3(self, obs, done):
        max_precision = 0.01    # 1cm
        max_reward = 1/max_precision
        scale = 0.1
        gripper_pos = obs[22:25]         # gripper x,y,z
        target_pos = obs[-3:]        # target x,y,z
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

    def init_or_load_weights(self):
        # --- actor
        # check if we need to load weights
        if self.path_to_model:
            print("Loading weights from %s to actor..." % self.path_to_model)
            self.actor.load_weights(os.path.join(self.path_to_model, "actor", "variables", "variables"))
            self.target_actor.load_weights(os.path.join(self.path_to_model, "actor", "variables", "variables"))
        else:
            # copy the weights to the target actor
            # update_target_variables(self.target_actor.weights, self.actor.weights, tau=1.0)
            self.target_actor.set_weights(self.actor.get_weights())
        # --- critic
        # check if we need to load weights
        if self.path_to_model:
            print("Loading weights from %s to critic..." % self.path_to_model)
            self.critic.load_weights(os.path.join(self.path_to_model, "critic", "variables", "variables"))
            self.target_critic.load_weights(os.path.join(self.path_to_model, "critic", "variables", "variables"))
        else:
            # copy the weights to the target critic
            # update_target_variables(self.target_critic.weights, self.critic.weights, tau=1.0)
            self.target_critic.set_weights(self.critic.get_weights())

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
        if self.replay_buffer_mode == "VANILLA":
            states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)
        elif self.replay_buffer_mode == "PER_PYTHON" or self.replay_buffer_mode == "PER_CPP":
            states, actions, rewards, next_states, dones, tree_idxs, is_weights = self.replay_buffer.sample_batch(self.batch_size)

        states = tf.constant(states)
        actions = tf.constant(actions)
        rewards = tf.constant(rewards)
        next_states = tf.constant(next_states)
        dones = tf.constant(dones)

        # call the inner function
        if self.replay_buffer_mode == "VANILLA":
            crit_loss, act_loss, td_errors = self.train_inner(states, actions, rewards, next_states, dones)
        elif self.replay_buffer_mode == "PER_PYTHON" or self.replay_buffer_mode == "PER_CPP":
            crit_loss, act_loss, td_errors = self.train_inner(states, actions, rewards, next_states, dones, is_weights)
            # update prioritized replay buffer
            self.replay_buffer.update_mult(tree_idxs, td_errors)

        return crit_loss, act_loss

    @tf.function
    def train_inner(self, states, actions, rewards, next_states, dones, is_weights=1.0):
        # --- Critic training ---
        with tf.GradientTape() as tape:
            td_errors = self._compute_td_error(states, actions, rewards, next_states, dones)
            td_errors_is = td_errors * is_weights

            # lets use a squared error for errors less than 1 and a linear error for errors greater than 1 to reduce
            # the impact of very large errors
            critic_loss = tf.reduce_mean(tf.square(td_errors_is))
            #td_errors = tf.abs(td_errors)
            #clipped_td_errors = tf.clip_by_value(td_errors, 0.0, 1.0)
            #linear_errors = 2 * (td_errors - clipped_td_errors)
            #critic_loss = tf.reduce_mean(tf.square(clipped_td_errors) + linear_errors)

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
        if self.use_target_copying:
            if self.global_step_main % self.interval_copy_target == 0 and self.global_step_main > 10:
                #update_target_variables(self.target_critic.weights, self.critic.weights, 1.0)
                #update_target_variables(self.target_actor.weights, self.actor.weights, 1.0)
                self.target_actor.set_weights(self.actor.get_weights())
                self.target_critic.set_weights(self.critic.get_weights())
        else:
            update_target_variables(self.target_critic.weights, self.critic.weights, self.tau)
            update_target_variables(self.target_actor.weights, self.actor.weights, self.tau)

        return critic_loss, actor_loss, td_errors

    def clean_up(self):
        print("Cleaning up ...")
        # shutdown all environments
        [q.put(("kill", ())) for q in self.command_queue]
        [worker.join() for worker in self.workers]
        # delete replay buffer (this safes the replay buffer one last time)
        del self.replay_buffer
