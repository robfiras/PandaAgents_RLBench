import numpy


class SumTree:
    """ binary tree structure used for prioritized experience replay. """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.max_prio_idx = None

    def _propagate(self, idx, change):
        """ updates the parent node given a child index and a change """
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """ given a (parent) index and a (random) variable s, the index of the child that lies in the priority range
         s < p_child_range <= s is returned -> if s is random in [0, p_total], this is equal to sampling a child
         according to its priority """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """ get the sum of all priorities in the buffer """
        return self.tree[0]

    def update(self, idx, p):
        """ update priority of given index and propagate value to root """
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """ get the index, priority and data index of a sample given a (random) variable s """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], data_idx
