import multiprocessing as mp

import numpy as np
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
from rlbench.environment import Environment
from rlbench.action_modes import ActionMode

from agents.misc.opts_arg_evaluator import eval_opts_args


class Agent(object):

    def __init__(self, action_mode: ActionMode, task_class, obs_config: ObservationConfig, argv):

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
        self.n_additional_workers = options["n_worker"]
        self.headless = options["headless"]

        self.action_mode = action_mode
        self.task_class = task_class
        self.obs_config = obs_config

        if not self.only_low_dim_obs:
            raise ValueError("High-dim observations currently not supported!")

        # multiprocessing stuff
        self.workers = []
        self.command_queue = []
        self.result_queue = []
        if self.n_additional_workers > 0:
            self.run_workers(self.n_additional_workers)

        # mur main thread is always running an environment as well
        self.env = Environment(action_mode=self.action_mode,
                               obs_config=self.obs_config,
                               headless=self.headless)

        self.env.launch()
        self.task = self.env.get_task(self.task_class)

        print('Initializing Episode')
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

    def run_workers(self, n_workers):
        self.command_queue = [mp.Queue()] * n_workers
        self.result_queue = [mp.Queue()] * n_workers
        self.workers = [mp.Process(target=self.job_worker,
                                   args=(worker_id,
                                         self.action_mode,
                                         self.obs_config,
                                         self.task_class,
                                         self.command_queue[worker_id],
                                         self.result_queue[worker_id])) for worker_id in range(n_workers)]
        for worker in self.workers:
            worker.start()

    def job_worker(self, worker_id, action_mode,
                    obs_config, task_class,
                    command_q: mp.Queue,
                    result_q: mp.Queue):
        env = Environment(action_mode=action_mode, obs_config=obs_config, headless=True)
        env.launch()
        task = env.get_task(task_class)
        task.reset()
        wait_for_reset = False
        print("Initialized worker %d" % worker_id)
        while True:
            command = command_q.get()
            command_type = command[0]
            command_args = command[1]
            if command_type == "reset":
                descriptions, observation = task.reset()
                result_q.put((descriptions, observation.get_low_dim_data()))
                wait_for_reset = False
            elif command_type == "step":
                actions = command_args[0]
                if not wait_for_reset:
                    next_observation, reward, done = task.step(actions)
                    result_q.put((next_observation, reward, done))
                    if done:
                        wait_for_reset = True
                else:
                    # wait until reset | we need to put something into the queue since the main process would not go on
                    result_q.put(None)
            elif command_type == "kill":
                print("Killing worker %d" % worker_id)
                env.shutdown()
                break






















