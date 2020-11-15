import random
from enum import Enum

from agents.ddpg_backend.replay_buffer import ReplayBuffer
from agents.ddpg_backend.sum_tree import SumTree
from sum_tree_cpp import SumTreeCpp

import numpy as np


class PrioReplayBufferType(Enum):
    CPP = 0
    PYTHON = 1


class PrioReplayBuffer(ReplayBuffer):

    def __init__(self, maxlen, dim_observations=None, dim_actions=None,
                 path_to_db_read=None, path_to_db_write=None, write=False, save_interval=None, use_cpp=False):
        # call parent constructor
        super(PrioReplayBuffer, self).__init__(maxlen, dim_observations, dim_actions,
                                               path_to_db_read, path_to_db_write, write, save_interval)

        # set type of prio buffer extension -> either C++ or Python
        if use_cpp:
            self.prio_buffer_type = PrioReplayBufferType.CPP
        else:
            self.prio_buffer_type = PrioReplayBufferType.PYTHON

        # init sum_tree storing all priorities
        if self.prio_buffer_type == PrioReplayBufferType.PYTHON:
            self.priority_tree = SumTree(self.maxlen)
        elif self.prio_buffer_type == PrioReplayBufferType.CPP:
            self.priority_tree = SumTreeCpp(self.maxlen)

        self.e = 0.001
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self.max_prio = 99.0

        # initialize priority tree if buffer was read
        if self.path_to_db_read:
            if self.prio_buffer_type == PrioReplayBufferType.PYTHON:
                self.priority_tree[0:self.length] = np.full(self.length, self.max_prio)
            elif self.prio_buffer_type == PrioReplayBufferType.CPP:
                self.priority_tree.init(self.max_prio)

    def append(self, state, action, reward, next_state, done, idx_episode):
        """ Appends the replay buffer with a single sample """
        data = np.concatenate((np.array([self.number_of_samples_seen]), np.array([idx_episode]), np.array(state),
                               np.array(action), np.array([reward]), np.array(next_state), np.array([done])))

        self.buf[self.index] = data

        # update priority in sum-tree
        tree_idx = self.index + self.maxlen - 1
        self.update(tree_idx, self.max_prio)

        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen
        self.number_of_samples_seen += 1

        # check if buffer needs to be saved
        if self.write and self.number_of_samples_seen % self.save_interval == 0:
            self.save_buffer()

    def sample(self, batch_size, with_replacement=True):
        tree_idxs = []
        data_idxs = []
        priorities = []
        segment = self.priority_tree.total() / batch_size

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            begin = segment * i
            end = segment * (i+1)

            s = random.uniform(begin, end)
            if self.prio_buffer_type == PrioReplayBufferType.PYTHON:
                idx, p, data_idx = self.priority_tree.get(s)
            elif self.prio_buffer_type == PrioReplayBufferType.CPP:
                p, data_idx = self.priority_tree.get(s)
                idx = data_idx + self.maxlen - 1

            tree_idxs.append(idx)
            data_idxs.append(data_idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.priority_tree.total()
        is_weights = np.power((1/self.length) * (1/sampling_probabilities), self.beta)
        is_weights /= is_weights.max()

        return self.buf[data_idxs], tree_idxs, is_weights

    def sample_batch(self, batch_size):
        """ Splits a sampled batch into states, actions, rewards, next_states and dones """
        data, tree_idxs, is_weights = self.sample(batch_size)
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
        return states, actions, rewards.reshape(-1, 1), next_states, dones.reshape(-1, 1), tree_idxs, is_weights

    def update(self, tree_idx, error):
        p = self.get_priority(error)
        self.max_prio = max(self.max_prio, p)
        if self.prio_buffer_type == PrioReplayBufferType.PYTHON:
            self.priority_tree.update(tree_idx, p)
        elif self.prio_buffer_type == PrioReplayBufferType.CPP:
            leaf_idx = tree_idx - self.maxlen + 1
            self.priority_tree.update(leaf_idx, p)

    def update_mult(self, tree_idxs, errors):
        priorities = self.get_priority(errors)
        if self.prio_buffer_type == PrioReplayBufferType.PYTHON:
            for i, p in zip(tree_idxs, priorities):
                self.priority_tree.update(i, p)
        elif self.prio_buffer_type == PrioReplayBufferType.CPP:
            leaf_idxs = np.array(tree_idxs) - self.maxlen + 1
            self.priority_tree.update_mult(leaf_idxs, priorities)

    def get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def save_buffer(self):
        """ Saves all samples in the replay buffer, which were appended since last saving """
        print("\nSaving replay buffer ...")
        start = self.last_saved_at % self.maxlen
        end = self.number_of_samples_seen % self.maxlen
        if start >= end:
            start = start - end
            end = self.maxlen
        data = self.buf[start:end]
        tree_start = start + self.maxlen - 1
        tree_end = end + self.maxlen - 1
        if self.prio_buffer_type == PrioReplayBufferType.PYTHON:
            priorities = self.priority_tree.tree[tree_start:tree_end]
        elif self.prio_buffer_type == PrioReplayBufferType.CPP:
            priorities = np.array(self.priority_tree.get_leaf_priorities(start, end))
        data = np.concatenate((data, priorities.reshape(len(priorities), 1)), axis=1)
        data_list_of_tuples = map(tuple, data.tolist())
        self.c.executemany('INSERT INTO data VALUES %s' % self.create_input_str(), data_list_of_tuples)
        self.conn.commit()
        self.last_saved_at = self.number_of_samples_seen
        print("Finished saving.")

    def create_input_str(self):
        """ Creates a string for inserting the buffer data into a sql table """
        input_str = "("
        for var in range(self.dim_one_sample):
            input_str += "?,"
        input_str += "?)"
        return input_str

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
        column_names += "Done float,"
        column_names += "Priority float"
        return column_names
