import multiprocessing as mp

import numpy as np
from rlbench.observation_config import ObservationConfig
from rlbench.environment import Environment
from rlbench.action_modes import ActionMode

from agents.base import Agent
import agents.misc.utils as utils


class RLAgent(Agent):

    def __init__(self, action_mode: ActionMode, obs_config: ObservationConfig, task_class, agent_config):

        # call parent constructor
        super(RLAgent, self).__init__(action_mode, obs_config, task_class, agent_config)

        # multiprocessing stuff
        self.workers = []
        self.command_queue = []
        self.result_queue = []
        self.n_workers = self.cfg["RLAgent"]["Setup"]["n_workers"]
        if self.n_workers > 1 and not self.headless:
            print("Turning headless mode on, since more than one worker is  running.")
            self.headless = True
        if self.save_weights:
            self.save_weights_interval = utils.adjust_save_interval(self.save_weights_interval, self.n_workers)
        if self.mode == "online_training":
            self.run_workers()
        self.worker_conn = [{"command_queue": cq,
                             "result_queue": rq,
                             "index": idx} for cq, rq, idx in zip(self.command_queue,
                                                                  self.result_queue,
                                                                  range(self.n_workers))]

    def run_workers(self):
        self.command_queue = [mp.Queue() for i in range(self.n_workers)]
        self.result_queue = [mp.Queue() for i in range(self.n_workers)]
        self.workers = [mp.Process(target=self.job_worker,
                                   args=(worker_id,
                                         self.action_mode,
                                         self.obs_config,
                                         self.task_class,
                                         self.command_queue[worker_id],
                                         self.result_queue[worker_id],
                                         self.obs_scaling_vector,
                                         self.headless)) for worker_id in range(self.n_workers)]
        for worker in self.workers:
            worker.start()

    def job_worker(self, worker_id, action_mode,
                   obs_config, task_class,
                   command_q: mp.Queue,
                   result_q: mp.Queue,
                   obs_scaling,
                   headless):

        np.random.seed(worker_id)
        env = Environment(action_mode=action_mode, obs_config=obs_config, headless=headless)
        env.launch()
        task = env.get_task(task_class)
        task.reset()
        print("Initialized worker %d" % worker_id)
        while True:
            command = command_q.get()
            command_type = command[0]
            command_args = command[1]
            if command_type == "reset":
                descriptions, observation = task.reset()
                if obs_scaling is not None:
                    observation = observation.get_low_dim_data() / obs_scaling
                else:
                    observation = observation.get_low_dim_data()
                result_q.put((descriptions, observation))
            elif command_type == "step":
                actions = command_args[0]
                next_observation, reward, done = task.step(actions)
                if obs_scaling is not None:
                    next_observation = next_observation.get_low_dim_data() / obs_scaling
                else:
                    next_observation = next_observation.get_low_dim_data()
                result_q.put((next_observation, reward, done))
            elif command_type == "kill":
                print("Killing worker %d" % worker_id)
                env.shutdown()
                break






















