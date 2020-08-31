import numpy as np
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
from rlbench.environment import Environment
from rlbench.action_modes import ActionMode


class Agent(object):

    def __init__(self, action_mode: ActionMode, task_class, obs_config: ObservationConfig, headless):

        self.obs_config = obs_config

        self.env = Environment(action_mode=action_mode,
                               obs_config=obs_config,
                               headless=headless)

        if not self.only_low_dim_obs:
            raise ValueError("High-dim observations currently not supported!")

        self.env.launch()
        self.task = self.env.get_task(task_class)

        print('Reset Episode')
        descriptions, obs = self.task.reset()
        print(descriptions)

        # determine dimensions
        self.dim_observations = np.shape(obs.get_low_dim_data())[0]    # TODO: find better way
        self.dim_actions = self.env.action_size

    @property
    def only_low_dim_obs(self) -> bool:
        """ Returns true, if only low-dim obs are set """

        low_dim_true = (self.obs_config.joint_velocities
                        and self.obs_config.joint_positions
                        and self.obs_config.joint_forces
                        and self.obs_config.gripper_open
                        and self.obs_config.gripper_pose
                        and self.obs_config.gripper_joint_positions
                        and self.obs_config.gripper_touch_forces
                        and self.obs_config.task_low_dim_state)

        high_dim_true = (self.obs_config.left_shoulder_camera.rgb
                         and self.obs_config.left_shoulder_camera.depth
                         and self.obs_config.left_shoulder_camera.mask
                         and self.obs_config.right_shoulder_camera.rgb
                         and self.obs_config.right_shoulder_camera.depth
                         and self.obs_config.right_shoulder_camera.mask
                         and self.obs_config.wrist_camera.rgb
                         and self.obs_config.wrist_camera.depth
                         and self.obs_config.wrist_camera.mask
                         and self.obs_config.front_camera.rgb
                         and self.obs_config.front_camera.depth
                         and self.obs_config.front_camera.mask)

        print("low_dim:", low_dim_true)
        print("high-dim: ", high_dim_true)

        if low_dim_true and not high_dim_true:
            return True
        else:
            return False
