import numpy as np
from rlbench.backend.observation import Observation
from rlbench.action_modes import ActionMode
from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, action_size):
        self.action_size = action_size

    # def act_old(self, obs):
    #     arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
    #     gripper = [1.0]  # Always open
    #     return np.concatenate([arm, gripper], axis=-1)

    @abstractmethod
    def act(self, action_mode: ActionMode, ):
        pass

