import os
import random
import time
import sys

import numpy as np
import tensorflow as tf

from agents.misc.logger import CmdLineLogger
from agents.misc.success_evaluator import SuccessEvaluator
from agents.misc.camcorder import Camcorder
import rlbench
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes import ActionMode
from rlbench.environment import Environment
from rlbench.sim2real.domain_randomization_environment import DomainRandomizationEnvironment
from rlbench.sim2real.domain_randomization import VisualRandomizationConfig, RandomizeEvery


class Agent(object):

    def __init__(self, action_mode: ActionMode, obs_config: ObservationConfig, task_class, agent_config):

        self.cfg = agent_config

        # setup some general parameters
        setup = self.cfg["Agent"]["Setup"]
        valid_modes = ["online_training", "offline_training", "validation", "validation_mult"]
        self.mode = setup["mode"]
        if self.mode not in valid_modes:
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
        self.scale_robot_observations = setup["scale_robot_observations"]
        self.save_camera_input = setup["save_camera_input"]
        self.logging_interval = setup["logging_interval"]
        if self.load_model_run_id:
            self.path_to_model = os.path.join(self.root_log_dir, self.load_model_run_id)
        else:
            self.path_to_model = None

        # setup rlbench environment
        self.action_mode = action_mode
        self.task_class = task_class
        self.obs_config = obs_config
        self.rand_env = setup["domain_randomization_environment"]
        if self.rand_env:
            rand_env_setup = setup["domain_randomization_setup"]
            if not rand_env_setup["image_directory"]:
                # use textures in rlbench's test folder
                rlbench_path = os.path.dirname(rlbench.__file__)
                # go one dir back
                RLBench_path = os.path.dirname(rlbench_path)
                # append dir to textures
                self.rand_texture_dir = os.path.join(RLBench_path, "tests", "unit", "assets", "textures")
                print("\nNo texture folder provided for DomainRandomization. Using default folder at %s." % self.rand_texture_dir)
            else:
                self.rand_texture_dir = rand_env_setup["image_directory"]
                if not os.path.exists(self.rand_texture_dir):
                    raise FileNotFoundError("Provided path %s does not exist!" % self.rand_texture_dir)
            self.visual_rand_config = VisualRandomizationConfig(image_directory=self.rand_texture_dir)
            self.randomize_every = rand_env_setup["randomize_every"]
            if self.randomize_every == "episode":
                self.randomize_every = RandomizeEvery.EPISODE
            elif self.randomize_every == "variation":
                self.randomize_every = RandomizeEvery.VARIATION
            elif self.randomize_every == "transition":
                self.randomize_every = RandomizeEvery.TRANSITION
            else:
                raise ValueError("%s is not a supported randomization mode." % self.randomize_every)
        else:
            self.visual_rand_config = None
            self.randomize_every = None

        # validation during training setup
        self.make_validation_during_training = setup["make_validation_during_training"]
        self.validation_interval = setup["validation_in_training_setup"]["validation_interval"]
        self.n_validation_episodes = setup["validation_in_training_setup"]["n_validation_episodes"]

        # set seed of random and numpy
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # set dimensions
        self.dim_observations = self.cfg["Robot"]["Dimensions"]["observations"]
        self.dim_actions = self.cfg["Robot"]["Dimensions"]["actions"]

        # check if we use a gripper action or not
        self.keep_gripper_open = self.cfg["Robot"]["keep_gripper_open"]

        # add an custom/unique id for logging
        if not self.root_log_dir:
            print("You have set the path to the root logging directory in the config-file.")
            self.root_log_dir = input("Please enter the path to the root logging directory:")
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
        # robot limits
        robot_limits_for_scaling = [self.gripper_open_limits, self.joint_vel_limits, self.joint_pos_limits_rad,
                                    self.joint_force_limits, self.gripper_pose_limits, self.gripper_joint_pos_limits,
                                    self.gripper_force_limits]
        robot_obs_config = [self.obs_config.gripper_open, self.obs_config.joint_velocities,
                            self.obs_config.joint_positions, self.obs_config.joint_forces, self.obs_config.gripper_pose,
                            self.obs_config.gripper_joint_positions, self.obs_config.gripper_touch_forces]
        filtered_limits = [limit for active, limit in zip(robot_obs_config, robot_limits_for_scaling) if active]

        # redundancy resolution setup
        self.use_redundancy_resolution = setup["use_redundancy_resolution"]
        self.redundancy_resolution_setup = setup["redundancy_resolution_setup"]
        self.redundancy_resolution_setup["ref_position"] = np.array(self.redundancy_resolution_setup["ref_position"])
        self.redundancy_resolution_setup["lower_joint_pos_limit"] = self.lower_joint_pos_limits_rad
        self.redundancy_resolution_setup["upper_joint_pos_limit"] = self.upper_joint_pos_limits_rad
        self.redundancy_resolution_setup["use_redundancy_resolution"] = self.use_redundancy_resolution

            # concatenate to scaling vector
        if self.scale_robot_observations:
            # only the robot state is scaled
            self.obs_scaling_vector = np.concatenate(filtered_limits)
            # append with ones for low-dim task state
            self.obs_scaling_vector = np.append(
                self.obs_scaling_vector, np.ones(self.dim_observations - len(self.obs_scaling_vector)))

        else:
            self.obs_scaling_vector = None

    def run_validation(self, model):
        """ Runs validation for presentation purposes
        :param model: Tensorflow model for action prediction
        """
        if self.rand_env:
            env = DomainRandomizationEnvironment(action_mode=self.action_mode, obs_config=self.obs_config,
                                                 headless=self.headless, randomize_every=self.randomize_every,
                                                 visual_randomization_config=self.visual_rand_config)
        else:
            env = Environment(action_mode=self.action_mode, obs_config=self.obs_config, headless=self.headless)
        env.launch()
        task = env.get_task(self.task_class)
        task.reset()
        print("Initialized Environment for validation")
        observation = None
        episode = 0
        step = 0
        done = False
        logger = CmdLineLogger(self.logging_interval, self.training_episodes, 1)
        evaluator = SuccessEvaluator()

        # containers for validation
        reward_per_episode = 0
        dones = 0
        rewards = 0
        episode_lengths = 0

        # we can save all enables camera inputs to root log dir if we want
        if self.save_camera_input:
            camcorder = Camcorder(self.root_log_dir, 0)
        while episode < self.training_episodes:
            # reset episode if maximal length is reached or all worker are done
            if step % self.episode_length == 0 or done:

                descriptions, observation = task.reset()

                if self.save_camera_input:
                    camcorder.save(observation, task.get_robot_visuals(), task.get_all_graspable_objects())
                if self.obs_scaling_vector is not None:
                    observation = observation.get_low_dim_data() / self.obs_scaling_vector
                else:
                    observation = observation.get_low_dim_data()
                logger(episode, evaluator.successful_episodes, evaluator.successful_episodes_perc)

                if step != 0:
                    rewards += reward_per_episode
                    episode_lengths += step

                step = 0
                reward_per_episode = 0
                episode += 1

            # predict action
            actions = np.squeeze(model.predict(tf.constant([observation])))
            actions[0:7] = actions[0:7]
            # check if we need to resolve redundancy
            if self.use_redundancy_resolution:
                actions[0:7], _ = task.resolve_redundancy_joint_velocities(actions=actions[0:7],
                                                                           setup=self.redundancy_resolution_setup)

            # check if we need to use a gripper actions
            if self.keep_gripper_open:
                actions[7] = 1

            # increment simulation
            next_observation, reward, done = task.step(actions, camcorder if self.save_camera_input else None)
            if self.obs_scaling_vector is not None:
                next_observation = next_observation.get_low_dim_data() / self.obs_scaling_vector
            else:
                next_observation = next_observation.get_low_dim_data()
            evaluator.add(episode, done)
            observation = next_observation
            step += 1
            reward_per_episode += reward
            dones += done

        env.shutdown()
        print("\nDone.\n")
        return rewards/self.episode_length, dones/self.training_episodes, episode_lengths/self.training_episodes

    def run_validation_post_sequential(self, model):

        path_to_validation_weights = os.path.join(self.root_log_dir, "weights_validation")

        if not os.path.exists(path_to_validation_weights):
            print("weights_validation path (%s) not found!" % path_to_validation_weights)
            sys.exit()

        # this should always run in headless mode
        self.headless = True

        # setup tensorboard
        summary_writer = tf.summary.create_file_writer(logdir=os.path.join(self.root_log_dir, "tb_valid_post"))

        # iterate over all weights in weights_validation
        for weights_dir in os.listdir(path_to_validation_weights):

            # get the weight id (training episode)
            weight_id = int(weights_dir.split(sep="_")[-1])

            # load weights in actor model
            model.load_weights(os.path.join(path_to_validation_weights, weights_dir, "actor", "variables", "variables"))

            # run validation of model
            print("Validation weight of training episode %d ..." % weight_id)
            avg_reward_per_episode, success_prop, avg_succes_episode_length = self.run_validation(model)
            print("Average reward: %f  | Proportion of successful episodes: %f | Average success episode length. %f" %
                  (avg_reward_per_episode, success_prop, avg_succes_episode_length))

            with summary_writer.as_default():
                tf.summary.scalar('Post Validation | Average Reward per Episode', avg_reward_per_episode,
                                  step=weight_id)

                tf.summary.scalar('Post Validation | Proportion of successful Episodes',
                                  success_prop,
                                  step=weight_id)

                tf.summary.scalar('Post Validation | Average Episode Length',
                                  avg_succes_episode_length,
                                  step=weight_id)

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
