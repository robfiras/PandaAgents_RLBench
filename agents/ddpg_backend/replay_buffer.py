import errno
import os
import sqlite3
import warnings
import io

import numpy as np


class ReplayBuffer(object):
    """
    This replay buffer stores all experiences in a deque and allows to randomly sample experiences from it.
    """

    def __init__(self, maxlen, dim_observations=None, dim_actions=None,
                 path_to_db=None, read=False, write=False, save_interval=None):
        self.maxlen = maxlen
        if not save_interval:
            self.save_interval = maxlen
        else:
            self.save_interval = min(save_interval, maxlen)
            if save_interval > maxlen:
                warnings.warn("You can not set the buffer's saving interval greater than the buffer size.")
        self.number_of_samples_seen = 0
        self.last_saved_at = 0
        self.index = 0
        self.length = 0
        self.write = write
        self.read = read
        self.path = path_to_db
        self.dim_observations = dim_observations
        self.dim_actions = dim_actions
        self.dim_one_sample = 2 * self.dim_observations + self.dim_actions + 2
        self.buf = np.empty(shape=(maxlen,self.dim_one_sample) , dtype=np.float)

        if (self.write or self.read) and not self.path:
            raise TypeError("You can not read or write data if not path to the database is defined.")

        if (self.dim_observations is None or self.dim_actions is None) and self.read:
            raise TypeError("You can not read, if no dimensions for the observations and actions are specified.")

        # check if the path to dir exists
        if self.write and self.path and not os.path.isdir(self.path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.path)

        # create connection to database and create if it does not exist yet
        self.conn = sqlite3.connect(os.path.join(self.path, 'replay_buffer.db'))
        self.c = self.conn.cursor()

        # create the table "data" of it does not exist
        sql_create_data_table = """ CREATE TABLE IF NOT EXISTS data (%s); """ % self.create_column_names()
        self.c.execute(sql_create_data_table)
        self.conn.commit()

    def __del__(self):
        # save the buffer one last time
        self.save()

    def append(self, state, action, reward, next_state, done):
        data = np.concatenate((np.array(state), np.array(action), np.array([reward]), np.array(next_state), np.array([done])))
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen
        self.number_of_samples_seen += 1
        # check if buffer needs to be saved
        if self.number_of_samples_seen % self.save_interval == 0:
            self.save()

    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size)  # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]

    def sample_batch(self, batch_size):
        data = self.sample(batch_size)
        ind = 0
        states = data[:, ind:self.dim_observations]
        ind += self.dim_observations
        actions = data[:, ind:(ind+self.dim_actions)]
        ind += self.dim_actions
        rewards = data[:, ind:(ind+1)]
        ind += 1
        next_states = data[:, ind:(ind+self.dim_observations)]
        ind += self.dim_observations
        dones = data[:, ind:(ind+1)]
        return states, actions, rewards.reshape(-1, 1), next_states, dones.reshape(-1, 1)

    def save(self):
        print("Saving replay buffer ...")
        start = self.last_saved_at % self.maxlen
        end = self.number_of_samples_seen % self.maxlen
        if start > end:
            start = start - end
            end = self.maxlen
        data = self.buf[start:end]
        data_list_of_tuples = map(tuple, data.tolist())
        self.c.executemany('INSERT INTO data VALUES %s' % self.create_input_str(), data_list_of_tuples)
        self.conn.commit()
        self.last_saved_at = self.number_of_samples_seen
        print("Finished saving.")

    def create_column_names(self):
        column_names = ""
        for var in range(self.dim_observations):
            column_names += "State_" + str(var) + " float,"
        for var in range(self.dim_actions):
            column_names += "Action_" + str(var) + " float,"
        column_names += "Reward float,"
        for var in range(self.dim_observations):
            column_names += "Next_State_" + str(var) + " float,"
        column_names += "Done float"
        return column_names

    def create_input_str(self):
        dim_one_sample = 2 * self.dim_observations + self.dim_actions + 2
        input_str = "("
        for var in range(dim_one_sample - 1):
            input_str += "?,"
        input_str += "?)"
        return input_str
