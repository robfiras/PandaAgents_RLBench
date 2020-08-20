# Copyright 2020 The handson-ml Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or    implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
import errno
import os


class ReplayBuffer(object):
    """
    This replay buffer stores all experiences in a deque and allows to randomly sample experiences from it.
    """

    def __init__(self, maxlen, store=False, path=None):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0
        self.store = store
        self.path = path

        if self.store and not self.path:
            raise TypeError("You can not store data if not path is defined.")

        # check if the path to dir exists
        if self.store and self.path and not os.path.isdir(self.path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.path)

    def append(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size)  # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]

    def sample_batch(self, batch_size):
        cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
        for memory in self.sample(batch_size):
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)