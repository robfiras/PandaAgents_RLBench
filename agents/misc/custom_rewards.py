import numpy as np

from rlbench.backend.observation import Observation


def euclidean_distance_reward(obs: Observation):
    """ Calculates the reward for the ReachTarget Task based on the euclidean distance.
        Observation is given as observation object. """
    max_precision = 0.01    # 1cm
    max_reward = 1/max_precision
    scale = 0.1
    gripper_pos = obs.gripper_pose[0:3]         # gripper x,y,z
    target_pos = obs.task_low_dim_state         # target x,y,z
    dist = np.sqrt(np.sum(np.square(np.subtract(target_pos, gripper_pos)), axis=0))     # euclidean norm
    reward = min((1/(dist + 0.00001)), max_reward)
    reward = scale * reward
    return reward


def euclidean_distance_reward_vec(obs: np.array):
    """ Calculates the reward for the ReachTarget Task based on the euclidean distance.
        Observation is given as a numpy array. """
    max_precision = 0.01    # 1cm
    max_reward = 1/max_precision
    scale = 0.1
    gripper_pos = obs[22:25]        # gripper x,y,z
    target_pos = obs[-3:]           # target x,y,z
    dist = np.sqrt(np.sum(np.square(np.subtract(target_pos, gripper_pos)), axis=0))     # euclidean norm
    reward = min((1/(dist + 0.00001)), max_reward)
    reward = scale * reward
    return reward
