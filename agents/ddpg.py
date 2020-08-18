
from agents.base import Agent
import numpy as np
from rlbench.backend.observation import Observation
from rlbench.action_modes import ActionMode
from abc import ABC, abstractmethod
import tensorflow as tf
from agents.ddpg_backend.replay_buffer import ReplayBuffer
from agents.ddpg_backend.target_update_ops import update_target_variables
from agents.ddpg_backend.ou_noise import OUNoise

#tf.config.experimental_run_functions_eagerly(True)

class ActorNetwork(tf.keras.Model):

    def __init__(self, units_hidden_layers, n_actions, activations=None, sigma=0.1, mu=0.0, use_ou_noise=False):
        # call the parent constructor
        super(ActorNetwork, self).__init__()

        # check if default activations should be used (relus)
        if not activations:
            activations = ["relu"] * len(units_hidden_layers)

        # define the layers that are going to be used in our actor
        self.hidden_layers = [tf.keras.layers.Dense(dim, activation=activation)
                              for dim, activation in zip(units_hidden_layers, activations)]
        self.out = tf.keras.layers.Dense(n_actions)

        # setup the noise for the actor
        self.sigma = sigma
        self.mu = mu
        # check if ou noise should be used
        self.__use_ou_noise = use_ou_noise
        if self.__use_ou_noise:
            self.__ou_noise = OUNoise(n_actions, mu=self.mu, sigma=self.sigma)

    def call(self, inputs):
        # define the forward pass
        z = inputs
        for layer in self.hidden_layers:
            z = layer(z)
        return self.out(z)

    def noisy_prediction(self, obs):
        # get prediction without noise
        pred = self.predict(obs)

        # add noise
        if self.__use_ou_noise:
            pred = self.add_ou_noise(pred)
        else:
            pred = self.add_gaussian_noise(pred)

        return pred

    def add_gaussian_noise(self, predictions):
        """ adds noise from a normal (Gaussian) distribution """
        noisy_predictions = predictions + tf.random.normal(shape=predictions.shape, mean=self.mu, stddev=self.sigma,
                                                           dtype=tf.float32)
        return noisy_predictions

    def add_ou_noise(self, predictions):
        """ adds noise from a Ornstein-Uhlenbeck process """
        noisy_predictions = predictions + self.__ou_noise.noise(predictions)
        return noisy_predictions


class CriticNetwork(tf.keras.Model):

    def __init__(self, units_hidden_layers, n_outputs, activations=None):
        # call the parent constructor
        super(CriticNetwork, self).__init__()

        # check if default activations should be used (relus)
        if not activations:
            activations = ["relu"] * len(units_hidden_layers)

        # define the layers that are going to be used in our critic
        self.hidden_layers = [tf.keras.layers.Dense(dim, activation=activation)
                              for dim, activation in zip(units_hidden_layers, activations)]
        self.out = tf.keras.layers.Dense(n_outputs)

    def call(self, inputs):
        # define the forward pass
        observations, actions = inputs
        z = tf.concat([observations, actions], axis=1)
        for layer in self.hidden_layers:
            z = layer(z)
        return self.out(z)


class DDPG(Agent):
    def __init__(self, n_actions,
                 buffer_size=50000,
                 lr_actor=0.0001,
                 lr_critic=0.001,
                 layers_actor=[400, 300],
                 layers_critic=[400,300],
                 gamma=0.9,
                 tau=0.001,
                 sigma=0.1
                 ):
        """
        :param n_actions: number of actions
        :param lr_actor: learning rate actor
        :param lr_critic: learning rate critic
        :param layers_actor: number of units in each dense layer in the actor (len of list defines number of layers)
        :param layers_critic: number of units in each dense layer in the actor (len of list defines number of layers)
        :param gamma: discount factor
        :param tau: hyperparameter soft target updates
        :param sigma: standard deviation for noise
        :param buffer_size: size of the replay buffer
        """

        self.n_actions = n_actions

        # setup the replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        #

        # --- define actor and its target---
        self.actor = ActorNetwork(layers_actor, n_actions)
        self.target_actor = ActorNetwork(layers_actor, n_actions)
        # copy the weights to the target actor
        update_target_variables(self.target_actor.weights, self.actor.weights, tau=1.0)
        # setup the actor's optimizer
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)

        # --- define the critic and its target ---
        self.critic = CriticNetwork(layers_critic, n_outputs=1) # one Q-value per state needed
        self.target_critic = CriticNetwork(layers_critic, n_outputs=1) # one Q-value per state needed
        # copy the weights to the target critic
        update_target_variables(self.target_critic.weights, self.critc.weights, tau=1.0)
        # setup the critic's optimizer
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)

        # setup the some hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma

    def act(self, obs: Observation):
        pass
        # print("these are the joint forces: ", obs.joint_forces)
        # print("these are the joint velocities: ", obs.joint_velocities)
        # print("these are the joint positions: ", obs.joint_positions)
        # print("this is the gripper pose: ", obs.gripper_pose)
        # print("this is the gripper_joint_pos: ", obs.gripper_joint_positions)
        # print("this is the gripper touch forces ", obs.gripper_touch_forces)
        # print("gripper open? ", obs.gripper_open)



actor = ActorNetwork([400, 300], 7)
ou_actor = ActorNetwork([400, 300], 7, use_ou_noise=True)
update_target_variables(ou_actor.weights, actor.weights, tau=1.0)

# TODO: update_target_variables scheint nicht zu funktionieren oder problem liegt wo anders
# TODO: step by step code checken
features = tf.constant([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]])
print("Ohne noise: ", actor.predict(features))
print("Mit noise: ", actor.noisy_prediction(features))

print("OU -> Ohne noise: ", ou_actor.predict(features))
print("OU -> Mit noise: ", ou_actor.noisy_prediction(features))

print("----- new -----")
noisee = OUNoise(7)
#print("state: ", noisee.state)
print("NOise :", noisee.noise(features).numpy())
#print("state: ", noisee.state)
