import sys

from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.reach_target import ReachTarget
import agents.misc.config_evaluator as config
from agents.td3 import TD3
from agents.openai_es import OpenAIES

# set the observation configuration
obs_config = ObservationConfig()

# use only low-dim observations
obs_only_low_dim = True     # currently only low-dim supported
obs_config.set_all_high_dim(not obs_only_low_dim)
obs_config.set_all_low_dim(obs_only_low_dim)

# define action mode
action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)

# (optional) read path to config-file from opts
path_config_file = config.get_path_to_config(sys.argv[1:])

print(path_config_file)

# create an agent
agent = TD3(action_mode=action_mode, obs_config=obs_config,
            task_class=ReachTarget, agent_config_path=path_config_file)

# run agent
agent.run()



