import multiprocessing as mp

import numpy as np
from rlbench.observation_config import ObservationConfig
from rlbench.environment import Environment
from rlbench.action_modes import ActionMode
from rlbench.sim2real.domain_randomization_environment import DomainRandomizationEnvironment

from agents.base import Agent
from agents.misc.camcorder import Camcorder
import agents.misc.utils as utils


class RLAgent(Agent):

    def __init__(self, action_mode: ActionMode, obs_config: ObservationConfig, task_class, agent_config):

        # call parent constructor
        super(RLAgent, self).__init__(action_mode, obs_config, task_class, agent_config)

        # multiprocessing stuff
        self.workers = []
        self.n_workers = self.cfg["RLAgent"]["Setup"]["n_workers"]
        self.command_queue = [mp.Queue() for i in range(self.n_workers)]
        self.result_queue = [mp.Queue() for i in range(self.n_workers)]
        if self.n_workers > 1 and not self.headless:
            print("Turning headless mode on, since more than one worker is  running.")
            self.headless = True
        if self.save_weights:
            self.save_weights_interval = utils.adjust_save_interval(self.save_weights_interval, self.n_workers)
        self.worker_conn = [{"command_queue": cq,
                             "result_queue": rq,
                             "index": idx} for cq, rq, idx in zip(self.command_queue,
                                                                  self.result_queue,
                                                                  range(self.n_workers))]

    def run_workers(self):
        self.workers = [mp.Process(target=job_worker,
                                   args=(worker_id,
                                         self.action_mode,
                                         self.obs_config,
                                         self.task_class,
                                         self.command_queue[worker_id],
                                         self.result_queue[worker_id],
                                         self.obs_scaling_vector,
                                         self.root_log_dir,
                                         self.save_camera_input,
                                         self.rand_env,
                                         self.visual_rand_config,
                                         self.randomize_every,
                                         self.headless)) for worker_id in range(self.n_workers)]
        for worker in self.workers:
            worker.start()


def job_worker(worker_id, action_mode,
               obs_config, task_class,
               command_q: mp.Queue,
               result_q: mp.Queue,
               obs_scaling,
               root_log_dir,
               save_camera_input,
               rand_env,
               visual_rand_config,
               randomize_every,
               headless):

    np.random.seed(worker_id)
    # setup the environment
    if rand_env:
        env = DomainRandomizationEnvironment(action_mode=action_mode, obs_config=obs_config,
                                             headless=headless, randomize_every=randomize_every,
                                             visual_randomization_config=visual_rand_config)
    else:
        env = Environment(action_mode=action_mode, obs_config=obs_config, headless=headless)
    env.launch()
    task = env.get_task(task_class)
    task.reset()
    # we can save all enables camera inputs to root log dir if we want
    if save_camera_input:
        camcorder = Camcorder(root_log_dir, worker_id)
    print("Initialized worker %d" % worker_id)
    while True:
        command = command_q.get()
        command_type = command[0]
        command_args = command[1]
        if command_type == "reset":
            descriptions, observation = task.reset()
            if save_camera_input:
                camcorder.save(observation, task.get_all_graspable_object_poses())
            if obs_scaling is not None:
                observation = observation.get_low_dim_data() / obs_scaling
            else:
                observation = observation.get_low_dim_data()
            result_q.put((descriptions, observation))
        elif command_type == "step":
            actions = command_args[0]
            next_observation, reward, done = task.step(actions)
            if save_camera_input:
                camcorder.save(next_observation, task.get_all_graspable_object_poses())
            if obs_scaling is not None:
                next_observation = next_observation.get_low_dim_data() / obs_scaling
            else:
                next_observation = next_observation.get_low_dim_data()
            result_q.put((next_observation, reward, done))
        elif command_type == "kill":
            print("Killing worker %d" % worker_id)
            env.shutdown()
            break






















