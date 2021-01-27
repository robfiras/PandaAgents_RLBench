import numpy as np

class SuccessEvaluator:
    def __init__(self, percentage_interval=500):
        self.successful_episodes = 0
        self.percentage_interval = percentage_interval
        self.successful_episodes_perc = 0.0
        self._episodes_in_interval = 0
        self._n_succ_last_interval = 0

        self._last_episode = 0

    def add(self, episode, done):
        if done:
            self._n_succ_last_interval += 1
            self.successful_episodes += 1
        if self._episodes_in_interval >= self.percentage_interval:
            self.successful_episodes_perc = self._n_succ_last_interval / self._episodes_in_interval
            self._episodes_in_interval = 0
            self._n_succ_last_interval = 0
        self._episodes_in_interval += episode - self._last_episode
        self._last_episode = episode

    def add_mult(self, episode, dones):
        n_dones = np.sum(dones)
        if n_dones > 0:
            self._n_succ_last_interval += n_dones
            self.successful_episodes += n_dones
        if self._episodes_in_interval >= self.percentage_interval:
            self.successful_episodes_perc = self._n_succ_last_interval / self._episodes_in_interval
            self._episodes_in_interval = 0
            self._n_succ_last_interval = 0
        self._episodes_in_interval += episode - self._last_episode
        self._last_episode = episode
