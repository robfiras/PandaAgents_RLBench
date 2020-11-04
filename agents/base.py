import os
import random
import time

import numpy as np
import tensorflow as tf

from agents.misc.opts_arg_evaluator import eval_opts_args
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes import ActionMode


class Agent(object):

    def __init__(self, action_mode: ActionMode, task_class, obs_config: ObservationConfig, argv, seed=94):

        # parse arguments
        self.argv = argv
        options = eval_opts_args(argv)
        self.root_log_dir = options["root_dir"]
        self.use_tensorboard = options["use_tensorboard"]
        self.save_weights = options["save_weights"]
        self.run_id = options["run_id"]
        self.path_to_model = options["path_to_model"]
        self.training_episodes = options["training_episodes"]
        self.no_training = options["no_training"]
        self.path_to_read_buffer = options["path_to_read_buffer"]
        self.write_buffer = options["write_buffer"]
        self.n_workers = options["n_worker"]
        self.headless = options["headless"]

        # non-headless mode only allowed, if one worker is set.
        if self.n_workers == 1 and not self.headless:
            self.headless = False
        else:
            self.headless = True

        self.action_mode = action_mode
        self.task_class = task_class
        self.obs_config = obs_config

        # set seed of random and numpy
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # set dimensions
        self.dim_observations = 40
        self.dim_actions = 8

        # add an custom/unique id for logging
        if (self.use_tensorboard or self.save_weights) and self.root_log_dir:
            if not self.run_id:
                self.run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
            self.root_log_dir = os.path.join(self.root_log_dir, self.run_id, "")

            # check if dir exists already
            if not os.path.exists(self.root_log_dir):
                print("\nCreating new directory: ", self.root_log_dir, "\n")
                os.mkdir(self.root_log_dir)

        # --- set observation limits for scaling
        # gripper open
        self.gripper_open_limits = np.array([1.0])
        # joint velocity limits
        self.joint_vel_limits = np.array([2.175, 2.175, 2.175, 2.175, 2.609, 2.609, 2.609])     # upper velocity limits taken from panda model
        # joint position -> max in pos/neg direction defines limit (for symmetric scaling)
        self.lower_joint_pos_limits_deg = [-166.0, -101.0, -166.0, -176.0, -166.0, -1.0, -166.0]  # min joint pos [°] taken from panda model
        self.range_joint_pos_deg = [332.0, 202.0, 332.0, 172.0, 332.0, 216.0, 332.0]    # range joint pos [°] taken from panda model
        self.lower_joint_pos_limits_rad = (np.array(self.lower_joint_pos_limits_deg)/360.0)*2*np.pi
        self.range_joint_pos_rad = (np.array(self.range_joint_pos_deg)/360.0)*2*np.pi
        self.upper_joint_pos_limits_rad = self.lower_joint_pos_limits_rad + self.range_joint_pos_rad
        self.joint_pos_limits_rad = np.amax((np.abs(self.lower_joint_pos_limits_rad), self.upper_joint_pos_limits_rad), axis=0)
        # joint forces
        self.joint_force_limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
        # gripper pose -> first 3 entry are x,y,z and rest are quaternions
        self.gripper_pose_limits = np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0])
        # gripper joint position
        self.gripper_joint_pos_limits = np.array([0.08, 0.08])
        # gripper touch forces -> 2* x,y,z
        self.gripper_force_limits = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
        # low dim task -> here for reach target
        self.low_dim_task_limits = np.array([3.0, 3.0, 3.0])
        # concatenate to scaling vector
        self.obs_scaling_vector = np.concatenate((self.gripper_open_limits,
                                                  self.joint_vel_limits,
                                                  self.joint_pos_limits_rad,
                                                  self.joint_force_limits,
                                                  self.gripper_pose_limits,
                                                  self.gripper_joint_pos_limits,
                                                  self.gripper_force_limits,
                                                  self.low_dim_task_limits))

        if not self.only_low_dim_obs:
            raise ValueError("High-dim observations currently not supported!")

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

        if low_dim_true and not high_dim_true:
            return True
        else:
            return False