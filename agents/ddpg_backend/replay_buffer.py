import errno
import os
import sqlite3
import warnings
from enum import Enum

import numpy as np


class ReplayBufferMode(Enum):
    # random sampling buffer
    VANILLA = 1

    # prioritized experience replay
    PER = 2

    # hindsight experience replay
    HER = 3


class ReplayBuffer(object):

    def __init__(self, maxlen, dim_observations=None, dim_actions=None,
                 path_to_db_read=None, path_to_db_write=None, write=False, save_interval=None):
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
        self.buffer_name = ""
        self.path_to_db_read = path_to_db_read
        self.path_to_db_write = path_to_db_write
        self.dim_observations = dim_observations
        self.dim_actions = dim_actions
        self.dim_one_sample = 2 * self.dim_observations + self.dim_actions + 4
        self.buf = np.empty(shape=(maxlen, self.dim_one_sample), dtype=np.float)

        if self.write and not self.path_to_db_write:
            raise TypeError("You can not write buffer data if not path to the database is defined.")

        if (self.dim_observations is None or self.dim_actions is None) and self.path_to_db_read:
            raise TypeError("You can not read, if no dimensions for the observations and actions are specified.")

        # check if the db exists before reading
        if self.path_to_db_read:
            self.path_to_db_read = os.path.join(self.path_to_db_read, self.__class__.__name__)
            if not os.path.exists(self.path_to_db_read):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.path_to_db_read)

        # read data into the buffer if needed
        if self.path_to_db_read:
            self.read_buffer(n_samples=maxlen)

        # check if we need to save the buffer during training 
        if self.write:
            self.path_to_db_write = os.path.join(self.path_to_db_write,  self.__class__.__name__)

            # create connection to database and create if it does not exist yet
            self.conn = sqlite3.connect(self.path_to_db_write)
            self.c = self.conn.cursor()

            # create the table "data" of it does not exist
            sql_create_data_table = """ CREATE TABLE IF NOT EXISTS data (%s); """ % self.create_column_names()
            self.c.execute(sql_create_data_table)
            self.conn.commit()

    def __del__(self):
        # save the buffer one last time and close connection
        if self.write:
            self.save_buffer()
            self.conn.close()

    def append(self, state, action, reward, next_state, done, idx_episode):
        """ Appends the replay buffer with a single sample """
        data = np.concatenate((np.array([self.number_of_samples_seen]), np.array([idx_episode]), np.array(state),
                               np.array(action), np.array([reward]), np.array(next_state), np.array([done])))
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen
        self.number_of_samples_seen += 1
        # check if buffer needs to be saved
        if self.write and self.number_of_samples_seen % self.save_interval == 0:
            self.save_buffer()

    def sample(self, batch_size, with_replacement=True):
        """ Samples a random batch of size batch_size from the replay buffer """
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size)  # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]

    def sample_batch(self, batch_size):
        """ Splits a randomly sampled batch into states, actions, rewards, next_states and dones """
        data = self.sample(batch_size)
        ind = 2     # first entry is sample_id and second is episode_id
        states = data[:, ind:(self.dim_observations+ind)]
        ind += self.dim_observations
        actions = data[:, ind:(ind+self.dim_actions)]
        ind += self.dim_actions
        rewards = data[:, ind:(ind+1)]
        ind += 1
        next_states = data[:, ind:(ind+self.dim_observations)]
        ind += self.dim_observations
        dones = data[:, ind:(ind+1)]
        return states, actions, rewards.reshape(-1, 1), next_states, dones.reshape(-1, 1)

    def save_buffer(self):
        """ Saves all samples in the replay buffer, which were appended since last saving """
        print("\nSaving replay buffer ...")
        start = self.last_saved_at % self.maxlen
        end = self.number_of_samples_seen % self.maxlen
        if start >= end:
            start = start - end
            end = self.maxlen
        data = self.buf[start:end]
        data_list_of_tuples = map(tuple, data.tolist())
        self.c.executemany('INSERT INTO data VALUES %s' % self.create_input_str(), data_list_of_tuples)
        self.conn.commit()
        self.last_saved_at = self.number_of_samples_seen
        print("Finished saving.")

    def read_buffer(self, n_samples):
        """ Returns the last n_samples samples from the replay buffer """
        print("\nReading Buffer from %s ..." % self.path_to_db_read)
        connection = sqlite3.connect(self.path_to_db_read)
        cursor = connection.cursor()
        column_names = self.get_column_names()
        input_str = "SELECT %s FROM (SELECT %s FROM data ORDER BY oid DESC Limit %d) ORDER BY RANDOM()" %\
                    (column_names, column_names, min(n_samples, self.maxlen))
        cursor.execute(input_str)
        data = np.array(cursor.fetchall())
        if data.shape[0] == 0:
            raise ValueError("Buffer is empty! Please check.")
        self.buf[0:data.shape[0], :] = data
        self.length = data.shape[0]
        self.number_of_samples_seen = self.length
        self.index = self.length - 1
        self.number_of_samples_seen = self.length
        connection.close()
        print("Finished reading buffer --> %d samples imported.\n" % self.length)

    def create_column_names(self):
        """ Creates a string for defining the columns in a table of a sql database """
        column_names = "Sample_ID int,"
        column_names += "Episode_ID int,"
        for var in range(self.dim_observations):
            column_names += "State_" + str(var) + " float,"
        for var in range(self.dim_actions):
            column_names += "Action_" + str(var) + " float,"
        column_names += "Reward float,"
        for var in range(self.dim_observations):
            column_names += "Next_State_" + str(var) + " float,"
        column_names += "Done float"
        return column_names

    def get_column_names(self):
        """ Creates a string for reading the columns in a table of a sql database
        Note: Priorities are not included! """
        column_names = "Sample_ID, "
        column_names += "Episode_ID, "
        for var in range(self.dim_observations):
            column_names += "State_" + str(var) + ", "
        for var in range(self.dim_actions):
            column_names += "Action_" + str(var) + ", "
        column_names += "Reward, "
        for var in range(self.dim_observations):
            column_names += "Next_State_" + str(var) + ", "
        column_names += "Done "
        return column_names

    def create_input_str(self):
        """ Creates a string for inserting the buffer data into a sql table """
        input_str = "("
        for var in range(self.dim_one_sample - 1):
            input_str += "?,"
        input_str += "?)"
        return input_str
