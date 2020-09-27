import tensorflow as tf


class TensorBoardLogger:
    def __init__(self, root_log_dir, percentage_interval=500):
        self.root_log_dir = root_log_dir
        self.summary_writer = tf.summary.create_file_writer(logdir=self.root_log_dir)
        self.number_of_succ_episodes = 0
        self.percentage_interval = percentage_interval
        self.n_succ_last_interval = 0
        self.percentage_succ = 0.0
        self.episodes_in_interval = 0
        self.last_episode = 0

    def __call__(self, total_steps, episode, losses, dones=None, step_in_episode=None,
                 rewards=None, epsilon=None, cond_train=False):
        with self.summary_writer.as_default():
            if cond_train:
                for name, value in losses.items():
                    if value:
                        tf.summary.scalar(name, value, step=total_steps)

            if rewards:
                tf.summary.scalar('Reward', sum(rewards), step=total_steps)

            if dones:
                for d in dones:
                    if d > 0.0:
                        self.number_of_succ_episodes += 1
                        self.n_succ_last_interval += 1
                        if step_in_episode:
                            tf.summary.scalar('Successful episode length', step_in_episode, step=total_steps)

            if epsilon:
                tf.summary.scalar('Epsilon', epsilon, step=total_steps)

            tf.summary.scalar('Number of successful Episodes', self.number_of_succ_episodes, step=total_steps)
            self.episodes_in_interval += episode - self.last_episode
            if self.episodes_in_interval >= self.percentage_interval:
                self.percentage_succ = self.n_succ_last_interval/self.episodes_in_interval
                tf.summary.scalar(('Proportion of successful episode in last %d episodes' % self.episodes_in_interval),
                                  self.percentage_succ, step=total_steps)
                self.episodes_in_interval = 0
                self.n_succ_last_interval = 0
            self.last_episode = episode
        return self.number_of_succ_episodes, self.percentage_succ
