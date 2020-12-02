import os
import multiprocessing as mp

import tensorflow as tf
import numpy as np

from rlbench.environment import Environment
from rlbench.sim2real.domain_randomization_environment import DomainRandomizationEnvironment
from agents.base_es import Network
from agents.misc.camcorder import Camcorder

# tf.config.run_functions_eagerly(True)
tf.keras.backend.set_floatx('float64')


class ESOptimizer(tf.Module):
    def __init__(self, model, learning_rate, momentum=0.9, name=None):
        super(ESOptimizer, self).__init__(name=name)
        weights = model.trainable_weights
        weight_shapes = [tf.shape(w) for w in weights]
        self.vs = [tf.Variable(tf.zeros(shape=shape, dtype=tf.float64)) for shape in weight_shapes]
        self.lr, self.momentum = tf.constant(learning_rate, dtype=tf.float64), tf.constant(momentum, dtype=tf.float64)

    def apply_gradients(self, weights,  cumulative_updates):
        for weight_tensor, v, cum_up in zip(weights, self.vs, cumulative_updates):
            v.assign(self.momentum * v + (1. - self.momentum) * cum_up)
            weight_tensor.assign_add(self.lr * v)


def job_descendant(descendant_id,
                   n_descendants_abs,
                   action_mode,
                   obs_config,
                   task_class,
                   command_q: mp.Queue,
                   result_q: mp.Queue,
                   reset_q: mp.Queue,
                   layers_network,
                   dim_actions,
                   max_actions,
                   obs_scaling_vector,
                   lr,
                   dim_observations,
                   utility,
                   lock: mp.Lock,
                   reward_shm: mp.Array,
                   finished_reward_writing,
                   finished_reward_reading,
                   start_reading_rewards,
                   seed,
                   noise_seeds,
                   sigma,
                   episode_length,
                   headless,
                   save_weights,
                   save_weights_interval,
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

    # define and instantiate one model for training and one for rollouts
    tf.random.set_seed(seed)  # same initial network for all descendants
    training_model = Network(layers_network, dim_actions, max_actions)
    training_model.build((1, dim_observations))
    # build() does not seem to be enough for instantiation -> make a dummy predict
    training_model.predict(tf.zeros(shape=(1, dim_observations)))
    rollout_model = Network(layers_network, dim_actions, max_actions)
    rollout_model.build((1, dim_observations))
    # build() does not seem to be enough for instantiation -> make a dummy predict
    rollout_model.predict(tf.zeros(shape=(1, dim_observations)))
    # copy the weights from the training model to the rollout model
    rollout_model.set_weights(training_model.get_weights())
    # setup an optimizer
    optimizer = ESOptimizer(rollout_model, lr)

    if path_to_model:
        print("\nReading model from ", path_to_model, "...\n")
        training_model.load_weights(os.path.join(path_to_model, "weights", "variables", "variables"))
        rollout_model.load_weights(os.path.join(path_to_model,  "weights", "variables", "variables"))

    # create a different Generator for each  descendant
    generators = [tf.random.Generator.from_seed(s) for s in noise_seeds]
    # create a single generator for perturbation in rollouts
    # this needed to call the above generators equally often to properly reconstruct the noise
    rollout_generator = tf.random.Generator.from_seed(noise_seeds[descendant_id])

    # we can save all enables camera inputs to root log dir if we want
    if save_camera_input:
        camcorder = Camcorder(root_log_dir, descendant_id)

    print("Initialized descendant %d" % descendant_id)

    signs = [sign(i) for i in range(n_descendants_abs)]

    episode = 0

    while True:
        command = command_q.get()
        if command == "run_episode_and_train":

            ''' 1. Perturbate the rollout network using the generator specific to this descendant '''
            perturbate_weights(rollout_model, rollout_generator, sigma, sign(descendant_id))

            ''' 2. Run an entire episode using the perturbated rollout network '''
            episode_reward = 0
            _, obs = task.reset()
            for i in range(episode_length):
                if save_camera_input:
                    camcorder.save(obs, task.get_all_graspable_object_poses())
                if obs_scaling_vector is None:
                    action = rollout_model.predict(tf.constant([obs.get_low_dim_data()]))
                else:
                    action = rollout_model.predict(tf.constant([obs.get_low_dim_data()/obs_scaling_vector]))
                obs, reward, done = task.step(np.squeeze(action))
                # return reward at the end of the episode
                if done:
                    episode_reward = reward
                    break
                elif i == (episode_length-1):
                    episode_reward = reward

            ''' 3. Add reward to reward shared memory to tell other workers about our reward and 
                read the rewards of other descendants as well '''
            # add our reward to shm
            lock.acquire()
            reward_shm[descendant_id] = episode_reward
            finished_reward_writing.value += 1
            if finished_reward_writing.value == n_descendants_abs:
                start_reading_rewards.set()
            result_q.put(done)
            lock.release()
            # wait for all threads to finish
            start_reading_rewards.wait()
            # collect rewards of all workers
            rewards = np.zeros(n_descendants_abs)
            lock.acquire()
            for i in range(n_descendants_abs):
                rewards[i] = reward_shm[i]
            finished_reward_reading.value += 1
            if finished_reward_reading.value == n_descendants_abs:
                reset_q.put("reset")
            lock.release()

            ''' 4. Now rank the descendants according to their reward and update the weights of the training network
                (which is not perturbated) by reconstructing each descendant's noise and determining its utilization
                based on the reward gathered in the last episode '''
            desc_rank = np.argsort(rewards)[::-1]
            cumulative_update(model=training_model,
                              generators=generators,
                              optimizer=optimizer,
                              rank=desc_rank,
                              utility=utility,
                              sigma=sigma,
                              signs=signs,
                              n_descendants_abs=n_descendants_abs)

            ''' 5. Copy the weights from the training model to the rollout model
                Note: All descendants have now the same training and rollout model weights again, because 
                all were updated with the same rewards and utilizations '''
            rollout_model.set_weights(training_model.get_weights())

            ''' 6. Check if we need to save weights | Only one descendant saves weights '''
            episode += n_descendants_abs
            if save_weights and (episode % save_weights_interval == 0) and descendant_id == 0:
                path_to_dir = os.path.join(root_log_dir, "weights", "")
                training_model.save(path_to_dir)

        elif command == "kill":
            print("Killing descendant %d" % descendant_id)
            env.shutdown()
            break
        else:
            raise ValueError("Sent command ", command, " for worker not supported!")


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


def sign(k_id):
    return -1. if k_id % 2 == 0 else 1.  # mirrored sampling


@tf.function
def perturbate_weights(model, gen, sigma, sign):
    weights = model.trainable_weights
    for weight_tensor in weights:
        weight_tensor.assign_add(sign * sigma * gen.normal(tf.shape(weight_tensor), dtype=tf.float64))


def cumulative_update(model, generators, optimizer, rank, utility, sigma, signs, n_descendants_abs):
    # sort generators and signs according to ranks
    weights = model.trainable_weights
    weight_shapes = [tf.shape(w) for w in weights]
    signs = np.array(signs)[rank]
    generators = np.array(generators)[rank]
    cumulative_updates = [tf.Variable(tf.zeros(shape=shape, dtype=tf.float64)) for shape in weight_shapes]

    # reconstruct noise and scale by utility
    for util, sign, gen in zip(utility, signs, generators):
        # iterate through weight vector to create cumulative updates
        for w_shape, cum_up in zip(weight_shapes, cumulative_updates):
            cum_up.assign_add(util * sign * sigma * gen.normal(shape=w_shape, dtype=tf.float64) / n_descendants_abs)

    # update the weights of the training model
    optimizer.apply_gradients(weights, cumulative_updates)










