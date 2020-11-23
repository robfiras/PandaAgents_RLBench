import os
import sys
import yaml

from rlbench.action_modes import ArmActionMode, GripperActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import reach_target, custom_pick_and_lift
from agents.ddpg import DDPG
from agents.td3 import TD3
from agents.openai_es import OpenAIES
import agents.misc.utils as utils


class AgentConfig:
    def __init__(self, agent_config_path=None):

        # read config file
        self.cfg = None
        if not agent_config_path:
            question = "No config-file path provided. Do you really want to continue with the default config-file?"
            if not utils.query_yes_no(question):
                print("Terminating ...")
                sys.exit()
            agent_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "default_config.yaml")
        with open(agent_config_path, "r") as stream:
            self.cfg = yaml.safe_load(stream)

        self.action_mode = self.__setup_action_mode()
        self.obs_config = self.__setup_obs_config()
        self.task_class = self.__setup_task_class()

    def __setup_action_mode(self):
        arm_action_mode_type = self.cfg["Agent"]["ArmActionMode"]
        arm_action_mode = None
        gripper_action_mode_type = self.cfg["Agent"]["GripperActionMode"]
        gripper_action_mode = None

        if arm_action_mode_type == "ABS_JOINT_VELOCITY":
            # Absolute arm joint velocities
            arm_action_mode = ArmActionMode.ABS_JOINT_VELOCITY

        elif arm_action_mode_type == "DELTA_JOINT_VELOCITY":
            # Change in arm joint velocities
            arm_action_mode = ArmActionMode.DELTA_JOINT_VELOCITY

        elif arm_action_mode == "ABS_JOINT_POSITION":
            # Absolute arm joint positions/angles (in radians)
            arm_action_mode = ArmActionMode.ABS_JOINT_VELOCITY

        elif arm_action_mode == "DELTA_JOINT_POSITION":
            # Change in arm joint positions/angles (in radians)
            arm_action_mode = ArmActionMode.DELTA_JOINT_POSITION

        elif arm_action_mode == "ABS_JOINT_TORQUE":
            # Absolute arm joint forces/torques
            arm_action_mode = ArmActionMode.ABS_JOINT_TORQUE

        elif arm_action_mode == "DELTA_JOINT_TORQUE":
            # Change in arm joint forces/torques
            arm_action_mode = ArmActionMode.DELTA_JOINT_TORQUE

        elif arm_action_mode == "ABS_EE_VELOCITY":
            # Absolute end-effector velocity (position (3) and quaternion (4))
            arm_action_mode = ArmActionMode.ABS_EE_VELOCITY

        elif arm_action_mode == "DELTA_EE_VELOCITY":
            # Change in end-effector velocity (position (3) and quaternion (4))
            arm_action_mode = ArmActionMode.DELTA_EE_VELOCITY

        elif arm_action_mode == "ABS_EE_POSE":
            # Absolute end-effector pose (position (3) and quaternion (4))
            arm_action_mode = ArmActionMode.ABS_EE_POSE

        elif arm_action_mode == "DELTA_EE_POSE":
            # Change in end-effector pose (position (3) and quaternion (4))
            arm_action_mode = ArmActionMode.DELTA_EE_POSE

        elif arm_action_mode == "ABS_EE_POSE_PLAN":
            # Absolute end-effector pose (position (3) and quaternion (4))
            # But does path planning between these points
            arm_action_mode = ArmActionMode.ABS_EE_POSE_PLAN

        elif arm_action_mode == "DELTA_EE_POSE_PLAN":
            # Change in end-effector pose (position (3) and quaternion (4))
            # But does path planning between these points
            arm_action_mode = ArmActionMode.DELTA_EE_POSE_PLAN

        else:
            raise ValueError(
                "%s is not a supported arm action mode. Please check your config-file." % arm_action_mode_type)

        if gripper_action_mode_type == "OPEN_AMOUNT":
            gripper_action_mode = GripperActionMode.OPEN_AMOUNT

        else:
            raise ValueError(
                "%s is not a supported gripper action mode. Please check your config-file." % gripper_action_mode_type)

        return ActionMode(arm=arm_action_mode, gripper=gripper_action_mode)

    def _get_camera_setup(self, name):
        if self.cfg["Agent"]["Observations"][name]:
            camera = self.cfg["CameraConfig"][name]

            return CameraConfig(rgb=camera["rgb"],
                                depth=camera["depth"],
                                mask=camera["mask"],
                                masks_as_one_channel=camera["mask_as_one_channel"],
                                image_size=(camera["image_size"][0], camera["image_size"][1]))
        else:
            camera_config = CameraConfig()
            camera_config.set_all(False)    # disable rgb, depth and mask
            return camera_config

    def __setup_obs_config(self):
        oc = self.cfg["Agent"]["Observations"]

        obs_config = ObservationConfig(left_shoulder_camera=self._get_camera_setup("left_shoulder_camera"),
                                       right_shoulder_camera=self._get_camera_setup("right_shoulder_camera"),
                                       wrist_camera=self._get_camera_setup("wrist_camera"),
                                       front_camera=self._get_camera_setup("front_camera"),
                                       joint_velocities=oc["joint_velocities"],
                                       joint_positions=oc["joint_positions"],
                                       joint_forces=oc["joint_forces"],
                                       gripper_open=oc["gripper_open"],
                                       gripper_pose=oc["gripper_pose"],
                                       gripper_joint_positions=oc["gripper_joint_positions"],
                                       gripper_touch_forces=oc["gripper_touch_forces"],
                                       task_low_dim_state=oc["task_low_dim_state"])

        return obs_config

    def __setup_task_class(self):
        task_class = self.cfg["Agent"]["Task"]
        if task_class == "ReachTarget":
            return reach_target.ReachTarget
        elif task_class == "PickAndLift":
            return custom_pick_and_lift.CustomPickAndLift
        else:
            raise ValueError("Until now only ReachTarget Task supported. More coming soon.")

    def get_agent(self):
        agent_type = self.cfg["Agent"]["Type"]
        if agent_type == "DDPG":
            return DDPG(self.action_mode, self.obs_config, self.task_class, self.cfg)
        elif agent_type == "TD3":
            return TD3(self.action_mode, self.obs_config, self.task_class, self.cfg)
        elif agent_type == "OpenAIES":
            return OpenAIES(self.action_mode, self.obs_config, self.task_class, self.cfg)
        else:
            raise ValueError("%s is not a supported agent type. Please check your config-file." % agent_type)
