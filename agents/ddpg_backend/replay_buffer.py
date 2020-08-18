from collections import deque
import random
import errno
import os


class ReplayBuffer(object):
    """
    This replay buffer stores all experiences in a deque and allows to randomly sample experiences from it.
    """

    def __init__(self, buffer_size, store=False, path=None):
        self.__buffer_size = buffer_size
        self.__curr_size = 0
        self.__buffer = deque()
        self.__store = store    # true if the data of the buffer should be stored
        self.__path = path

        if self.__store and not self.__path:
            raise TypeError("You can not store data if not path is defined.")

        # check if the path to dir exists
        if not os.path.isdir(self.__path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.__path)

    @property
    def size(self):
        return self.__buffer_size

    @property
    def curr_size(self):
        return self.__cur_size

    def sample_batch(self, batch_size):
        """
        :param batch_size: Size of the batch to be sampled
        :return: Batch of randomly sampled experiences from the buffer
        """
        return random.sample(self.__buffer, batch_size)

    def append(self, state, action, reward, next_state, done):
        """
        Stores a Tuple consisting of the current state, the action, the reward the subsequent state and a boolean
        telling if the subsequent state is a terminal state or not.
        """
        experience = (state, action, reward, next_state, done)
        if self.__curr_size < self.buffer_size:
            self.__buffer.append(experience)
            self.__curr_size += 1
        else:
            self.__buffer.popleft()
            self.__buffer.append(experience)

    def clear_buffer(self):
        self.__buffer = deque()
        self.__curr_size = 0
