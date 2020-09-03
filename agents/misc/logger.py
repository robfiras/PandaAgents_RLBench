import time
import datetime


class CmdLineLogger:
    def __init__(self, logging_interval, training_episodes, n_additional_workers=0, normalization=50):
        if logging_interval < (1+n_additional_workers):
            self.logging_interval = (1+n_additional_workers)
            print("\nLengthening the logging interval from %d to %d.\n" % (logging_interval, self.logging_interval))
        elif logging_interval % (1+n_additional_workers) != 0:
            self.logging_interval = logging_interval - (logging_interval % (1+n_additional_workers))
            print("\nShortening the logging interval from %d to %d.\n" % (logging_interval, self.logging_interval))
        else:
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
            print('%d.Episode - Status %.2f %% |%s%s| Number of successful Episodes %d | Time taken for the last %d Episodes: %f sec. | Aprox. Time needed till %s \r' % (
                    episode, (episode / self.training_episodes)*100, '#' * norm_status,
                    "." * (self.normalization - norm_status - 1), number_of_succ_episodes, self.logging_interval,
                    elapsed_time, str(eta)), end="")

            self.start = time.time()


