import os
import tensorflow as tf
import numpy as np
from agents.misc.success_evaluator import SuccessEvaluator


class TensorBoardLogger:
    def __init__(self, root_log_dir, percentage_interval=500):
        self.root_log_dir = os.path.join(root_log_dir, "tb_train", "")
        self.summary_writer = tf.summary.create_file_writer(logdir=self.root_log_dir)
        self.evaluator = SuccessEvaluator()

    def __call__(self, total_steps, episode, losses, dones=None, red_loss=None, step_in_episode=None, rewards=None, epsilon=None):
        with self.summary_writer.as_default():
            for name, value in losses.items():
                if value:
                    tf.summary.scalar(name, value, step=total_steps)

            if rewards:
                tf.summary.scalar('Reward', np.mean(rewards), step=total_steps)

            if dones:
                for d in dones:
                    self.evaluator.add(episode, d)
                    if d > 0.0 and step_in_episode:
                        tf.summary.scalar('Successful episode length', step_in_episode, step=total_steps)

            if red_loss:
                tf.summary.scalar('Loss Redundancy Resolution', np.mean(red_loss), step=total_steps)

            if epsilon:
                tf.summary.scalar('Epsilon', epsilon, step=total_steps)

            tf.summary.scalar('Number of successful Episodes', self.evaluator.successful_episodes, step=total_steps)
            tf.summary.scalar(('Proportion of successful episode in last %d episodes' % self.evaluator.percentage_interval),
                              self.evaluator.successful_episodes_perc, step=total_steps)

        return self.evaluator.successful_episodes, self.evaluator.successful_episodes_perc


class TensorBoardLoggerValidation:
    def __init__(self, root_log_dir):
        self.root_log_dir = os.path.join(root_log_dir, "tb_valid", "")
        self.summary_writer = tf.summary.create_file_writer(logdir=self.root_log_dir)

    def __call__(self, curr_training_episode, n_validation_episodes,
                 avg_reward_per_episode, n_success_episodes, avg_red_loss, avg_episode_length):
        with self.summary_writer.as_default():
            tf.summary.scalar('Validation | Average Reward per Episode', avg_reward_per_episode,
                              step=curr_training_episode)

            tf.summary.scalar('Validation | Number of successful Episodes', n_success_episodes,
                              step=curr_training_episode)

            tf.summary.scalar('Validation | Proportion of successful Episodes', n_success_episodes/n_validation_episodes,
                              step=curr_training_episode)

            tf.summary.scalar('Validation | Average Episode Length',
                              avg_episode_length,
                              step=curr_training_episode)

            tf.summary.scalar('Validation | Average Loss Redundancy Resolution',
                              avg_red_loss,
                              step=curr_training_episode)
