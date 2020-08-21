from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.reach_target import ReachTarget
from rlbench.environment import Environment
from agents.ddpg import DDPG

# set the observation configuration
obs_config = ObservationConfig()

# use only low-dim observations
obs_only_low_dim = True     # currently only low-dim supported
obs_config.set_all_high_dim(not obs_only_low_dim)
obs_config.set_all_low_dim(obs_only_low_dim)

# define action mode and environment
action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(action_mode=action_mode, obs_config=obs_config, headless=False)

# create an agent
agent = DDPG(env, ReachTarget, obs_config)

# define number of training steps
training_episodes = 100

# run agent
agent.run(training_episodes)



