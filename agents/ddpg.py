from agents.base import Agent
import numpy as np
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
import tensorflow as tf
from agents.ddpg_backend.replay_buffer import ReplayBuffer
from agents.ddpg_backend.target_update_ops import update_target_variables
from agents.ddpg_backend.ou_noise import OUNoise

# tf.config.experimental_run_functions_eagerly(True)
tf.keras.backend.set_floatx('float64')


class ActorNetwork(tf.keras.Model):

    def __init__(self, units_hidden_layers, dim_actions, activations=None, sigma=0.1, mu=0.0, use_ou_noise=False):
        # call the parent constructor
        super(ActorNetwork, self).__init__()

        # check if default activations should be used (relus)
        if not activations:
            activations = ["relu"] * len(units_hidden_layers)

        # define the layers that are going to be used in our actor
        self.hidden_layers = [tf.keras.layers.Dense(dim, activation=activation)
                              for dim, activation in zip(units_hidden_layers, activations)]
        self.out = tf.keras.layers.Dense(dim_actions, activation="tanh")

        # setup the noise for the actor
        self.sigma = sigma
        self.mu = mu
        # check if ou noise should be used
        self.use_ou_noise = use_ou_noise
        if self.use_ou_noise:
            self.ou_noise = OUNoise(dim_actions, mu=self.mu, sigma=self.sigma)

    def call(self, inputs):
        # define the forward pass
        z = inputs
        for layer in self.hidden_layers:
            z = layer(z)
        return self.out(z)

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
                                                           dtype=tf.float32)
        return noisy_predictions

    def add_ou_noise(self, predictions):
        """ adds noise from a Ornstein-Uhlenbeck process """
        noisy_predictions = predictions + self.ou_noise.noise(predictions)
        return noisy_predictions


class CriticNetwork(tf.keras.Model):

    def __init__(self, units_hidden_layers, dim_obs,  dim_outputs, activations=None):
        # call the parent constructor
        super(CriticNetwork, self).__init__()

        self.dim_obs = dim_obs

        # check if default activations should be used (relus)
        if not activations:
            activations = ["relu"] * len(units_hidden_layers)

        # define the layers that are going to be used in our critic
        self.hidden_layers = [tf.keras.layers.Dense(dim, activation=activation)
                              for dim, activation in zip(units_hidden_layers, activations)]
        self.out = tf.keras.layers.Dense(dim_outputs)

    def call(self, inputs):
        # define the forward pass
        z = inputs
        for layer in self.hidden_layers:
            z = layer(z)
        return self.out(z)


class DDPG(Agent):
    def __init__(self, env,
                 task_class,
                 obs_config,
                 gamma=0.9,
                 tau=0.001,
                 sigma=0.1,
                 batch_size=32,
                 episode_length=40,
                 training_interval=1,
                 start_training=0,
                 use_ou_noise=False,
                 buffer_size=50000,
                 lr_actor=0.0001,
                 lr_critic=0.001,
                 layers_actor=[400, 300],
                 layers_critic=[400, 300]
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
        :param use_ou_noise: if true, Ornstein-Uhlenbeck noise is used instead of Gaussian noise
        :param buffer_size: size of the replay buffer
        :param lr_actor: learning rate actor
        :param lr_critic: learning rate critic
        :param layers_actor: number of units in each dense layer in the actor (len of list defines number of layers)
        :param layers_critic: number of units in each dense layer in the actor (len of list defines number of layers)
        """

        # call parent constructor
        super(DDPG, self).__init__(env, task_class, obs_config)

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
        self.global_step = 0
        self.global_episode = 0
        self.use_ou_noise = use_ou_noise

        # setup the replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # --- define actor and its target---
        self.actor = ActorNetwork(layers_actor, self.dim_actions, sigma=sigma, use_ou_noise=use_ou_noise)
        self.target_actor = ActorNetwork(layers_actor,  self.dim_actions)
        # instantiate the models (if we do not instantiate the model, we can not copy their weights)
        self.actor.build((1, self.dim_inputs_actor))
        self.target_actor.build((1, self.dim_inputs_actor))
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
        # copy the weights to the target critic
        update_target_variables(self.target_critic.weights, self.critic.weights, tau=1.0)
        # setup the critic's optimizer
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)

    def run(self, training_episodes):
        obs = None
        while self.global_episode < training_episodes:
            if self.global_step % self.episode_length == 0:
                print('Reset Episode')
                descriptions, obs = self.task.reset()
                obs = obs.get_low_dim_data()
                print(descriptions)
                self.global_episode += 1

            #print("This is the obs:", obs)
            #print("Input dim is: ", self.dim_inputs_actor)
            #print("here are the limits: ", self.task.get_joint_upper_velocity_limits())
            # predict action with actor
            action = self.get_action([obs])
            #print("This is the action vector: ", action)

            # make a step
            next_obs, reward, done = self.task.step(action)
            next_obs = next_obs.get_low_dim_data()

            # add experience to replay buffer
            self.replay_buffer.append(obs, action, float(reward), next_obs, float(done))

            # train if conditions are met
            if self.global_step >= self.start_training and self.global_step % self.training_interval == 0:
                print("Training")
                self.train()

            # increment and save next_obs as obs
            self.global_step += 1
            obs = next_obs

        print('Done')
        self.env.shutdown()

    def get_action(self, obs, noise=True, scale=True):
        """
        Predicts an action using the actor network. DO NOT USE WHILE TRAINING. Use predict() or noisy_predict() instead
        :param obs: received observation
        :param noise: if true adds noise to actions tensor
        :param scale: if true scales actions | ONLY FOR ACTION_MODE ABS_JOINT_VELOCITY
        :return: returns an numpy array containing the corresponding actions
        """
        if noise:
            actions = self.actor.noisy_predict(tf.constant(obs))
        else:
            actions = self.actor.predict(tf.constant(obs))
        # squeeze since only one obs was given
        actions = np.squeeze(actions)
        if scale:
            actions = self.scale_action(actions)
        return actions

    def scale_action(self, actions):
        # in rad/s
        max_vel_joints = self.task.get_joint_upper_velocity_limits()
        max_vel_joints.append(1.0)  # add scaling for gripper (no scaling, action is discretized anyway)
        max_vel_joints = np.array(max_vel_joints)
        # action are between -1 and 1, so scaling is done by multiplication
        actions = actions * max_vel_joints

        # the gripper only accepts actions between 0 and 1 | clipping needed due to noise
        actions[-1] = np.clip(0.5*actions[-1] + 0.5, 0, 1)

        return actions

    @tf.function
    def _compute_td_error(self, states, actions, rewards, next_states, dones):
        not_dones = 1.0 - dones
        input_target_critic = tf.concat([next_states, actions], axis=1)
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

        @tf.function
        def train_inner():
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

        # call the inner function
        crit_loss, act_loss = train_inner()

        # TODO: log to tensorboard












# state = [0.34, 0.343]
# action = [0.34, 0.343, 0.34, 0.343,0.34, 0.343,0.34, 0.343,0.34]
# reward = 0.22
# next_state = [0.34, 0.343]
# done = 1
#
# agent = DDPG(2,9)
#
# agent.replay_buffer.append(state, action, reward, next_state, done)
#
# state = [0.3524, 0.343]
# action = [0.65, 0.88343, 0.665834, 0.343,0.565, 0.56,0.34, 0.3543,0.34]
# reward = 0.22
# next_state = [0.174, 0.345473]
# done = 0.0
#
# agent.replay_buffer.append(state, action, reward, next_state, done)
# agent.train()
#

#print(agent.get_action([[0.98, 0.47], [0.25, 0.2]]))

"""
actor = ActorNetwork([400, 300], 7)
ou_actor = ActorNetwork([400, 300], 7, use_ou_noise=True)

# update_target_variables(ou_actor.weights, actor.weights, tau=1.0)


actor.build((1, 2))
ou_actor.build((1, 2))

update_target_variables(ou_actor.weights, actor.weights, tau=1.0)

features = tf.constant([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]])
print("Ohne noise: ", actor.predict(features))
print("Mit noise: ", actor.noisy_predict(features))

print("OU -> Ohne noise: ", ou_actor.predict(features))
print("OU -> Mit noise: ", ou_actor.noisy_predict(features))

"""