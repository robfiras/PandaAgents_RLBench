import sys

from agents.agent_setup import AgentConfig
import agents.misc.config_evaluator as config

# (optional) read path to config-file from opts
# can be set manually as well
path_config_file = config.get_path_to_config(sys.argv[1:])

# setup agent configuration
agent_setup = AgentConfig(path_config_file)

# create an agent
agent = agent_setup.get_agent()

# run agent
agent.run()



