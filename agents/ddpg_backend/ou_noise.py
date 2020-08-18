# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/floodsung/DDPG/blob/master/ou_noise.py
# --------------------------------------
# Modifications copyright (C) 2020 Firas Al-Hafez
# adapted class to work with TF.2.X and varying batch sizes
import tensorflow as tf


class OUNoise:
    """ Creates temporally correlated noise """
    def __init__(self,  n_actions, mu=0.0, theta=0.15, sigma=0.2):
        self.n_actions = n_actions
        self.mu = tf.constant(mu)
        self.theta = tf.constant(theta)
        self.sigma = tf.constant(sigma)
        self.state = tf.Variable(tf.ones(self.n_actions) * self.mu)
        self.reset()

    def reset(self):
        self.state = tf.Variable(tf.ones(self.n_actions) * self.mu)

    def noise(self, actions):
        """ Returns a Tensor of OU Noise with the same shape the actions """

        # determine the number of samples
        n_samples = tf.shape(actions)[0]  # shape of axis=0
        # create empty variable for noise
        ou_noise = tf.Variable(tf.zeros([n_samples, self.n_actions]))

        @tf.function
        def noise_inner():
            for samples in tf.range(n_samples):
                x = self.state
                dx = self.theta * (self.mu - x) + self.sigma * tf.random.normal([self.n_actions])
                self.state.assign(self.state + dx)
                ou_noise[samples].assign(self.state)
            return ou_noise

        return noise_inner()
