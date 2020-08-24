import time
import datetime


class CmdLineLogger:
    def __init__(self, logging_interval, training_episodes, normalization=50):
        self.logging_interval = logging_interval
        self.training_episodes = training_episodes
        self.normalization = normalization
        self.start = time.time()

    def __call__(self, episode, number_of_succ_episodes):
        if episode % self.logging_interval == 0:
            norm_status = int(episode / (self.training_episodes / self.normalization))
            elapsed_time = time.time() - self.start
            eta = ((self.training_episodes - episode) / self.logging_interval) * elapsed_time
            eta = datetime.timedelta(seconds=eta)
            print(
                '%d.Episode - Status %.2f %% |%s%s| Number of successful Episodes %d | Time taken for the last %d Episodes: %f sec. | Aprox. Time needed till end %s\r' % (
                    episode, (episode / self.training_episodes)*100, "█" * norm_status,
                    "." * (self.normalization - norm_status - 1), number_of_succ_episodes, self.logging_interval, elapsed_time, str(eta)), end=' ')
            self.start = time.time()
