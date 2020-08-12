from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.reach_target import ReachTarget
from agent import Agent
import numpy as np
import time


# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(
    action_mode, DATASET, obs_config, False)
env.launch()

task = env.get_task(ReachTarget)
#demos = task.get_demos(2, live_demos=live_demos)

agent = Agent(env.action_size)
#agent.ingest(demos)

print("This is the action space size: ", env.action_size )


training_steps = 120
episode_length = 40
obs = None
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)
    action = agent.act(obs)
    print(action)
    obs, reward, terminate = task.step(action)
    print("These are the joint-velocities obs: ", obs.joint_velocities)
    print("These are the low-dim obs", obs.task_low_dim_state)
    time.sleep(10)


print('Done')
env.shutdown()
