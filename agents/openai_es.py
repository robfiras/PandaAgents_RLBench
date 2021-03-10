import os
import sys
import math
import time
import multiprocessing as mp
from copy import deepcopy

import numpy as np

from agents.base_es import ESAgent, Network
from agents.misc.logger import CmdLineLogger
from agents.misc.success_evaluator import SuccessEvaluator
from agents.openai_es_backend.es_utils import job_worker, tb_logger_job, tb_logger_job_validation
import agents.misc.utils as utils


class OpenAIES(ESAgent):

    def __init__(self,
                 action_mode,
                 obs_config,
                 task_class,
                 agent_config):

        # call parent constructor
        super(OpenAIES, self).__init__(action_mode, obs_config, task_class, agent_config)

        self.worker_seeds = np.random.randint(0, 2 ** 32 - 1, size=self.n_workers, dtype=np.uint32)

        # multiprocessing setup
        self.lock = mp.Lock()
        self.command_queues = [mp.Queue() for i in range(self.n_workers)]
        self.result_queues = [mp.Queue() for i in range(self.n_workers)]
        self.workers = [mp.Process(target=job_worker, args=(i,
                                                            self.action_mode,
                                                            self.obs_config,
                                                            self.task_class,
                                                            self.command_queues[i],
                                                            self.result_queues[i],
                                                            self.es_hparams,
                                                            self.dim_actions,
                                                            self.max_actions,
                                                            self.keep_gripper_open,
                                                            self.obs_scaling_vector,
                                                            self.dim_observations,
                                                            self.seed,
                                                            self.worker_seeds[i],
                                                            self.episode_length,
                                                            self.headless,
                                                            self.save_weights,
                                                            self.root_log_dir,
                                                            self.save_camera_input,
                                                            self.rand_env,
                                                            self.visual_rand_config,
                                                            self.randomize_every,
                                                            self.redundancy_resolution_setup,
                                                            self.path_to_model)) for i in range(self.n_workers)]

        # we need to setup a separate Tensorboard thread since tf can not be
        # imported in the main thread due to multiprocessing
        if self.use_tensorboard:
            self.tb_queue = mp.Queue()
            self.tb_logger = mp.Process(target=tb_logger_job, args=(self.root_log_dir,
                                                                    self.tb_queue))
        if self.make_validation_during_training:
            self.tb_queue_validation = mp.Queue()
            self.tb_logger_validation = mp.Process(target=tb_logger_job_validation, args=(self.root_log_dir,
                                                                                          self.tb_queue_validation))

    def run(self):
        if self.mode == "online_training":
            self.run_online_training()
        elif self.mode == "validation":
            validation_model = Network(self.es_hparams["layers_network"], self.dim_actions, self.max_actions)
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
        logger = CmdLineLogger(self.perturbations_per_batch, self.training_episodes, self.n_workers)
        evaluator = SuccessEvaluator()

        # start all threads
        for worker in self.workers:
            worker.daemon = True
            worker.start()
        if self.use_tensorboard:
            self.tb_logger.start()
        if self.make_validation_during_training:
            self.tb_logger_validation.start()

        episode = 0

        while episode < self.training_episodes:

            # trigger rollouts in threads
            env_seed = episode
            for q in self.command_queues:
                q.put(("run_n_episodes", (int(self.perturbations_per_batch/self.n_workers), env_seed)))

            # collect results
            results = []
            for q in self.result_queues:
                results.append(q.get())

            # flatten results
            results = np.array([item for worker_result in results for item in worker_result])
            
            # send the results to each worker and let them train
            for q in self.command_queues:
                q.put(("train", deepcopy(results)))

            # wait for workers to finish
            for q in self.result_queues:
                q.get()

            episode += self.perturbations_per_batch

            # check if we need to make validation
            validation_condition = (self.make_validation_during_training and
                                    episode % self.validation_interval == 0)
            if validation_condition:
                print("\nValidating ...")
                start_time = time.time()
                episode_to_validate_per_worker = math.ceil(self.n_validation_episodes / self.n_workers)
                episodes_left = self.n_validation_episodes
                for q in self.command_queues:
                    q.put(("validate", (min(episodes_left, episode_to_validate_per_worker), episode)))
                    episodes_left -= episode_to_validate_per_worker

                # collect results
                results_validation = []
                for q in self.result_queues:
                    results_validation.append(q.get())
                results_validation = np.array(results_validation)
                dones = np.sum(results_validation[:, 0])
                rewards = np.sum(results_validation[:, 1])
                avg_reward_per_episode = rewards / self.n_validation_episodes

                # log to commandline
                val_duration = time.time() - start_time
                print("Proportion of successful episodes %f and average reward per episode %f | Duration %f sec.\n" %
                      (dones / self.n_validation_episodes,
                       avg_reward_per_episode, val_duration))

                # log validation to tensorboard
                self.tb_queue_validation.put(("log", (episode, self.n_validation_episodes,
                                                      avg_reward_per_episode, dones)))

            # log to tensorboard if needed
            if self.use_tensorboard:
                mean_reward = np.mean(results[:, 2])
                dones = results[:, 3]
                evaluator.add_mult(episode, dones)
                self.tb_queue.put(("log", (episode, mean_reward, evaluator)))

            # log information to cmd-line
            logger(episode, evaluator.successful_episodes, evaluator.successful_episodes_perc)

        self.clean_up()
        print('\nDone.\n')

    def clean_up(self):
        print("Cleaning up ...")
        # shutdown all environments
        [q.put(("kill", None)) for q in self.command_queues]
        # shutdown TensorBoard logger
        if self.use_tensorboard:
            self.tb_queue.put(("kill", ()))
        if self.make_validation_during_training:
            self.tb_queue_validation.put(("kill", ()))
        [worker.join() for worker in self.workers]






