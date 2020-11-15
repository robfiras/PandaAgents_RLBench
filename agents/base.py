import os
import random
import time
import sys

import numpy as np
import tensorflow as tf

from agents.misc.logger import CmdLineLogger
from agents.misc.success_evaluator import SuccessEvaluator
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes import ActionMode
from rlbench.environment import Environment


class Agent(object):

    def __init__(self, action_mode: ActionMode, obs_config: ObservationConfig, task_class, agent_config):

        self.cfg = agent_config

        # setup some general parameters
        setup = self.cfg["Agent"]["Setup"]
        valid_modes = ["online_training", "offline_training", "validation"]
        if setup["mode"] in valid_modes:
            self.mode = setup["mode"]
        else:
            raise ValueError("Mode %s not supported." % self.mode)
        self.root_log_dir = setup["root_log_dir"]
        self.use_tensorboard = setup["use_tensorboard"]
        self.save_weights = setup["save_weights"]
        self.save_weights_interval = setup["save_weights_interval"]
        self.run_id = setup["run_id"]
        self.load_model_run_id = setup["load_model_run_id"]
        self.headless = setup["headless"]
        self.seed = setup["seed"]
        self.training_episodes = setup["episodes"]
        self.episode_length = setup["episode_length"]
        self.scale_observations = setup["scale_observations"]
        self.scale_actions = setup["scale_actions"]
        self.logging_interval = setup["logging_interval"]
        if self.load_model_run_id:
            self.path_to_model = os.path.join(self.root_log_dir, self.load_model_run_id)
        else:
            self.path_to_model = None

        self.action_mode = action_mode
        self.task_class = task_class
        self.obs_config = obs_config

        # set seed of random and numpy
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # set dimensions
        self.dim_observations = self.cfg["Robot"]["Dimensions"]["observations"]
        self.dim_actions = self.cfg["Robot"]["Dimensions"]["actions"]

        # add an custom/unique id for logging
        if (self.use_tensorboard or self.save_weights) and self.root_log_dir:
            if not self.run_id:
                self.run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
            self.root_log_dir = os.path.join(self.root_log_dir, self.run_id, "")

            # check if dir exists already
            if not os.path.exists(self.root_log_dir):
                print("\nCreating new directory: ", self.root_log_dir, "\n")
                os.mkdir(self.root_log_dir)

        # --- set observation and action limits for scaling
        robot_limits = self.cfg["Robot"]["Limits"]
        # max actions
        self.max_actions = robot_limits["actions"]
        # gripper open
        self.gripper_open_limits = np.array(robot_limits["gripper_open"])
        # joint velocity limits
        self.joint_vel_limits = np.array(robot_limits["joint_vel"])
        # joint position -> max in pos/neg direction defines limit (for symmetric scaling)
        self.lower_joint_pos_limits_deg = robot_limits["lower_joint_pos"]
        self.range_joint_pos_deg = robot_limits["range_joint_pos"]
        self.lower_joint_pos_limits_rad = (np.array(self.lower_joint_pos_limits_deg)/360.0)*2*np.pi
        self.range_joint_pos_rad = (np.array(self.range_joint_pos_deg)/360.0)*2*np.pi
        self.upper_joint_pos_limits_rad = self.lower_joint_pos_limits_rad + self.range_joint_pos_rad
        self.joint_pos_limits_rad = np.amax((np.abs(self.lower_joint_pos_limits_rad), self.upper_joint_pos_limits_rad), axis=0)
        # joint forces
        self.joint_force_limits = np.array(robot_limits["joint_force"])
        # gripper pose -> first 3 entry are x,y,z and rest are quaternions
        self.gripper_pose_limits = np.array(robot_limits["gripper_pose"])
        # gripper joint position
        self.gripper_joint_pos_limits = np.array(robot_limits["gripper_pos"])
        # gripper touch forces -> 2* x,y,z
        self.gripper_force_limits = np.array(robot_limits["gripper_force"])
        # low dim task -> here for reach target
        self.low_dim_task_limits = np.array(robot_limits["low_dim_task"])
        # concatenate to scaling vector
        if self.scale_observations:
            self.obs_scaling_vector = np.concatenate((self.gripper_open_limits,
                                                      self.joint_vel_limits,
                                                      self.joint_pos_limits_rad,
                                                      self.joint_force_limits,
                                                      self.gripper_pose_limits,
                                                      self.gripper_joint_pos_limits,
                                                      self.gripper_force_limits,
                                                      self.low_dim_task_limits))
        else:
            self.obs_scaling_vector = None

        if not self.only_low_dim_obs:
            raise ValueError("High-dim observations currently not supported!")

    def run_validation(self, model):
        """ Runs validation for presentation purposes
        :param model: Tensorflow model for action prediction
        """
        env = Environment(action_mode=self.action_mode, obs_config=self.obs_config, headless=False)
        env.launch()
        task = env.get_task(self.task_class)
        task.reset()
        print("Initialized Environment for validation")
        observation = None
        episode = 0
        step = 0
        logger = CmdLineLogger(self.logging_interval, self.training_episodes, 1)
        evaluator = SuccessEvaluator()
        while episode < self.training_episodes:
            # reset episode if maximal length is reached or all worker are done
            if step % self.episode_length == 0:
                descriptions, observation = task.reset()
                if self.obs_scaling_vector is not None:
                    observation = observation.get_low_dim_data() / self.obs_scaling_vector
                else:
                    observation = observation.get_low_dim_data()
                logger(episode, evaluator.successful_episodes, evaluator.successful_episodes_perc)
                step = 0
                episode += 1

            actions = np.squeeze(model.predict(tf.constant([observation])))
            next_observation, reward, done = task.step(actions)
            evaluator.add(episode, done)
            observation = next_observation.get_low_dim_data()
            step += 1
        env.shutdown()
        print("\nDone.\n")

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
