import os
import sys
import multiprocessing as mp

import numpy as np

from agents.base_es import ESAgent, Network
from agents.misc.logger import CmdLineLogger
from agents.misc.success_evaluator import SuccessEvaluator
from agents.openai_es_backend.es_utils import job_descendant, tb_logger_job
import agents.misc.utils as utils


class OpenAIES(ESAgent):

    def __init__(self,
                 action_mode,
                 obs_config,
                 task_class,
                 agent_config):

        # call parent constructor
        super(OpenAIES, self).__init__(action_mode, obs_config, task_class, agent_config)

        # setup utilization
        rank = np.arange(1, self.n_descendants_abs+1)
        util = np.maximum(0, np.log(self.n_descendants_abs/2 + 1) - np.log(rank))
        utility = util/util.sum() - 1/self.n_descendants_abs
        self.noise_seeds = np.random.randint(0, 2 ** 32 - 1, size=self.n_descendants, dtype=np.uint32)
        self.noise_seeds = np.repeat(self.noise_seeds, 2, axis=0).tolist()  # mirrored sampling

        # multiprocessing setup
        self.lock = mp.Lock()
        self.command_queues = [mp.Queue() for i in range(self.n_descendants_abs)]
        self.result_queues = [mp.Queue() for i in range(self.n_descendants_abs)]
        self.reset_queue = mp.Queue()
        self.reward_shm = mp.Array('d', [0.0]*self.n_descendants_abs)   # shared memory for reward communication between workers
        self.finished_reward_writing = mp.Value('i', 0)
        self.finished_reward_reading = mp.Value('i', 0)
        self.start_reading_rewards = mp.Event()
        self.descendants = [mp.Process(target=job_descendant, args=(i,
                                                                    self.n_descendants_abs,
                                                                    self.action_mode,
                                                                    self.obs_config,
                                                                    self.task_class,
                                                                    self.command_queues[i],
                                                                    self.result_queues[i],
                                                                    self.reset_queue,
                                                                    self.layers_network,
                                                                    self.dim_actions,
                                                                    self.max_actions,
                                                                    self.obs_scaling_vector,
                                                                    self.lr,
                                                                    self.dim_observations,
                                                                    utility,
                                                                    self.lock,
                                                                    self.reward_shm,
                                                                    self.finished_reward_writing,
                                                                    self.finished_reward_reading,
                                                                    self.start_reading_rewards,
                                                                    self.seed,
                                                                    self.noise_seeds,
                                                                    self.sigma,
                                                                    self.episode_length,
                                                                    self.headless,
                                                                    self.save_weights,
                                                                    self.save_weights_interval,
                                                                    self.root_log_dir,
                                                                    self.path_to_model)) for i in range(self.n_descendants_abs)]

        # we need to setup a separate Tensorboard thread since tf can not be
        # imported in the main thread due to multiprocessing
        if self.use_tensorboard:
            self.tb_queue = mp.Queue()
            self.tb_logger = mp.Process(target=tb_logger_job, args=(self.root_log_dir,
                                                                    self.tb_queue))

    def run(self):
        if self.mode == "online_training":
            self.run_online_training()
        elif self.mode == "validation":
            validation_model = Network(self.layers_network, self.dim_actions, self.max_actions)
            validation_model.build((1, self.dim_observations))
            if not self.path_to_model:
                question = "You have not set a path to model. Do you really want to validate a random model?"
                if not utils.query_yes_no(question):
                    print("Terminating ...")
                    sys.exit()
            else:
                print("\nReading model from ", self.path_to_model, "...\n")
                validation_model.load_weights(os.path.join(self.path_to_model,  "weights", "variables", "variables"))
            self.run_validation(validation_model)
        else:
            raise ValueError("\n%s mode not supported in OpenAI-ES.\n")

    def run_online_training(self):
        logger = CmdLineLogger(self.logging_interval, self.training_episodes, self.n_descendants_abs)
        evaluator = SuccessEvaluator()

        # start all threads
        for descendant in self.descendants:
            descendant.daemon = True
            descendant.start()
        if self.use_tensorboard:
            self.tb_logger.start()

        while self.global_episode < self.training_episodes:

            # trigger rollouts in threads
            for q in self.command_queues:
                q.put("run_episode_and_train")

            # wait for the descendants to rollout an episode, writing their reward
            # and reading the reward of other descendants
            reset_command = self.reset_queue.get()
            if reset_command == "reset":
                self.lock.acquire()
                # reset the counters
                self.finished_reward_reading.value = 0
                self.finished_reward_writing.value = 0
                # collect rewards and dones
                rewards = [0.0]*self.n_descendants_abs
                dones = [0.0]*self.n_descendants_abs
                for i in range(self.n_descendants_abs):
                    rewards[i] = self.reward_shm[i]
                    dones[i] = self.result_queues[i].get()
                self.lock.release()

                # log to tensorboard if needed
                if self.use_tensorboard:
                    for r, d in zip(rewards, dones):
                        evaluator.add(self.global_episode, d)
                        self.tb_queue.put(("log", (self.global_episode, r, evaluator)))
                        self.global_episode += 1
                else:
                    self.global_episode += self.n_descendants_abs

                # log information to cmd-line
                logger(self.global_episode, evaluator.successful_episodes, evaluator.successful_episodes_perc)

            else:
                raise ValueError("Received improper reset command: ", reset_command)

        self.clean_up()
        print('\nDone.\n')

    def clean_up(self):
        print("Cleaning up ...")
        # shutdown all environments
        [q.put("kill") for q in self.command_queues]
        # shutdown TensorBoard logger
        if self.use_tensorboard:
            self.tb_queue.put(("kill", ()))
        [worker.join() for worker in self.descendants]






