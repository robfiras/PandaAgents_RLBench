import os
import multiprocessing as mp
from copy import deepcopy

import tensorflow as tf
import numpy as np

from rlbench.environment import Environment
from rlbench.sim2real.domain_randomization_environment import DomainRandomizationEnvironment
from agents.base_es import Network
from agents.misc.camcorder import Camcorder

#tf.config.run_functions_eagerly(True)
#tf.keras.backend.set_floatx('float64')


class ESOptimizer(tf.Module):
    def __init__(self, weight_shapes, learning_rate, momentum=0.9):
        super(ESOptimizer, self).__init__()
        self.vs = [tf.Variable(tf.zeros(shape=shape, dtype=tf.float64), dtype=tf.float64) for shape in weight_shapes]
        self.lr, self.momentum = tf.constant(learning_rate, dtype=tf.float64), tf.constant(momentum, dtype=tf.float64)

    def apply_gradients(self, weights, cumulative_updates):
        for weight_tensor, v, cum_up in zip(weights, self.vs, cumulative_updates):
            v.assign(self.momentum * v + (1. - self.momentum) * cum_up)
            weight_tensor.assign_add(self.lr * v)

    def get_gradients(self, cumulative_updates):
        grads = [tf.Variable(tf.zeros_like(v)) for v in self.vs]
        for v, cum_up, grad in zip(self.vs, cumulative_updates, grads):
            v.assign(self.momentum * v + (1. - self.momentum) * cum_up)
            grad.assign(self.lr * v)
        return grads


def job_worker(worker_id,
               action_mode,
               obs_config,
               task_class,
               command_q: mp.Queue,
               result_q: mp.Queue,
               layers_network,
               dim_actions,
               max_actions,
               obs_scaling_vector,
               lr,
               dim_observations,
               common_seed,
               individual_seed,
               sigma,
               return_mode,
               episode_length,
               headless,
               save_weights,
               root_log_dir,
               save_camera_input,
               rand_env,
               visual_rand_config,
               randomize_every,
               path_to_model):

    # do not allow to run on GPU (does not work for multiple processes at the same time)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

    # each worker gets a different environment -> controlled by numpy seed 
    np.random.seed(individual_seed)

    # define and instantiate a model with same parameters by using the same tf seed for all workers 
    tf.random.set_seed(common_seed)  
    model = Network(layers_network, dim_actions, max_actions)
    model.build((1, dim_observations))
    # build() does not seem to be enough for instantiation -> make a dummy predict
    model.predict(tf.zeros(shape=(1, dim_observations)))

    # each worker has an unique generator
    generator = tf.random.Generator.from_seed(int(individual_seed), alg="philox")

    # save the network parameters --> these need to stay the same for all worker
    weights = model.get_weights()
    weight_shapes = [tf.shape(w) for w in weights]

    # setup an optimizer
    optimizer = ESOptimizer(weight_shapes, lr)

    if path_to_model:
        print("\nReading model from ", path_to_model, "...\n")
        model.load_weights(os.path.join(path_to_model, "weights", "variables", "variables"))

    # we can save all enables camera inputs to root log dir if we want
    if save_camera_input:
        camcorder = Camcorder(root_log_dir, worker_id)

    print("Initialized worker %d" % worker_id)

    while True:
        command, data = command_q.get()
        if command == "run_n_episodes":

            # data in this command is number of episodes the descendant needs to run
            n_episodes_worker = data

            if n_episodes_worker % 2 != 0:
                raise ValueError("The number of episodes per descendant needs to be dividable by 2 as we use "
                                 "mirrored sampling!")

            results_episodes = []
            episode = 0
            while episode < n_episodes_worker:

                # save the current state of the generator
                state_generator = deepcopy(generator.state)

                """ 1. Run one episode with positive perturbations """
                sign = 1

                # set the weights of the model
                model.set_weights(weights)

                # perturbate the weights of the model
                perturbate_weights(model, generator, sigma, sign)

                # run one episode with the perturbated model
                episode_return, was_done_once = run_one_episode(model, task, episode_length, obs_scaling_vector,
                                                                camcorder if save_camera_input else None)

                # save the current state of the generator, the sign, the cumulated reward and the done
                results_episodes.append((state_generator, sign, episode_return, was_done_once))

                """ 2. Run one episode with negative perturbations """
                sign = -1

                # set the weights of the model
                model.set_weights(weights)

                # reset the state of the generator to be the same as for the positive perturbation
                generator.reset(state_generator)

                # perturbate the weights of the model
                perturbate_weights(model, generator, sigma, sign)

                # run one episode with the perturbated model
                episode_return, was_done_once = run_one_episode(model, task, episode_length, obs_scaling_vector,
                                                                camcorder if save_camera_input else None)

                # save the current state of the generator, the sign, the cumulated reward and the done
                results_episodes.append((state_generator, sign, episode_return, was_done_once))

                episode += 2

            # send the results to the master
            result_q.put(results_episodes)

        elif command == "train":

            # data in this command are the results from all workers
            results_from_all_workers = data
            generator_states = results_from_all_workers[:, 0]
            signs = results_from_all_workers[:, 1]
            rewards = results_from_all_workers[:, 2]

            # calculate returns depending on the return mode
            if return_mode == "plain_rewards":
                returns = rewards
            elif return_mode == "utility_function":
                rank_range = np.arange(0, len(rewards))
                util = np.maximum(0, np.log(len(rewards) / 2 + 1) - np.log(rank_range))
                returns = util / util.sum() - 1 / len(rewards)
                # order returns, signs and generator_states by rewards
                desc_rank = np.argsort(rewards)[::-1]
                signs = signs[desc_rank]
                generator_states = generator_states[desc_rank]
            elif return_mode == "centered_ranks":
                rank_range = np.arange(0, len(rewards))
                returns = np.flip(rank_range / rank_range[-1] - 0.5)
                # order returns by rewards
                desc_rank = np.argsort(rewards)[::-1]
                signs = signs[desc_rank]
                generator_states = generator_states[desc_rank]
            else:
                raise ValueError("Return mode not supported yet.")

            # update weights
            gradients = get_cumulative_grads(weight_shapes, returns, generator_states, signs, sigma, optimizer)
            for i in range(len(weights)):
                weights[i] += gradients[i]

            # set the weights to the model
            model.set_weights(weights)

            # save weights
            if save_weights:
                path_to_dir = os.path.join(root_log_dir, "weights", "")
                model.save(path_to_dir)

        elif command == "kill":
            print("Killing worker %d" % worker_id)
            env.shutdown()
            break
        else:
            raise ValueError("Sent command ", command, " not supported!")


def run_one_episode(model, task, episode_length, obs_scaling_vector, camcorder):
    episode_return = 0.0
    was_done_once = 0
    _, obs = task.reset()
    for i in range(episode_length):
        if camcorder:
            camcorder.save(obs, task.get_all_graspable_object_poses())
        if obs_scaling_vector is None:
            action = model.predict(tf.constant([obs.get_low_dim_data()]))
        else:
            action = model.predict(tf.constant([obs.get_low_dim_data() / obs_scaling_vector]))
        obs, reward, done = task.step(np.squeeze(action))
        # cumulate reward
        episode_return += reward
        was_done_once |= done
    return episode_return, was_done_once


def tb_logger_job(root_log_dir, tb_queue):

    import tensorflow as tf

    summary_writer = tf.summary.create_file_writer(logdir=root_log_dir)
    while True:

        command, data = tb_queue.get()
        if command == "log":
            episode, reward, evaluator = data
            with summary_writer.as_default():
                tf.summary.scalar('Reward', reward, step=episode)
                tf.summary.scalar('Number of successful Episodes', evaluator.successful_episodes,
                                  step=episode)
                tf.summary.scalar(('Proportion of successful episode in last %d episodes' %
                                   evaluator.percentage_interval),
                                  evaluator.successful_episodes_perc, step=episode)
        elif command == "kill":
            print("Killing TensorBoard Logger")
            break
        else:
            raise ValueError("Sent command ", command, " for TensorBoard Logger not supported!")


@tf.function
def perturbate_weights(model, gen, sigma, sign):
    weights = model.trainable_weights
    for weight_tensor in weights:
        noise = gen.normal(tf.shape(weight_tensor), dtype=tf.float64)
        weight_tensor.assign_add(sign * sigma * noise)


def get_cumulative_grads(weights_shapes, returns, generator_states, signs, sigma, optimizer):

    cumulative_updates = [tf.Variable(tf.zeros(shape=shape, dtype=tf.float64), dtype=tf.float64) for shape in weights_shapes]
    batch_size = len(returns)

    # reconstruct noise
    for r, gen_state, sign in zip(returns, generator_states, signs):
        # recover noise vector from generator state
        generator = tf.random.Generator.from_state(gen_state, alg="philox")
        # iterate over weight vector to create cumulative update
        for up, shape in zip(cumulative_updates, weights_shapes):
            noise = generator.normal(shape=shape, dtype=tf.float64)
            up.assign_add(r * sign * sigma * noise / batch_size)

    # update the weights of the training model
    grads = optimizer.get_gradients(cumulative_updates)

    return grads





