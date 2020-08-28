# PandaAgents_RLBench
This project aims at creating reinforcement learning agents and evolution strategies tailored towards [RLBench](https://github.com/stepjam/RLBench).
These agents are designed to allow quick benchmarking with respect to your own algorithms on the RLBench tasks. Moreover, it is planned to add a ROS-API to allow seamless training in the real world as well. At the time being, all agents are designed for the Franka Emika Panda. All agents are built with Tensorflow 2.0.  


## Installation 
1. Download and install  [CoppeliaSim](https://www.coppeliarobotics.com/downloads), [PyRep](https://github.com/stepjam/PyRep) and [RLBench](https://github.com/robfiras/RLBench) (forked from https://github.com/stepjam/RLBench),

2. Install all required packages:

```shell 
pip install -r requirements.txt
```

3. Clone this Repository to your desired location:

```shell 
git clone https://github.com/robfiras/PandaAgents_RLBench
```

*Note:* This version is tested on Python 3.7.7. Currently it does not work with Python 2 or Python 3.8 or above.

## Currently included Agents and Features
**Agents**:
- Deep Deterministic Policy Gradient (DDPG)

### In Progress:
- Including a ROS-API

## Planned Agents and Features
**Agents**:
- PI² 
- CMA-ES
- OpenAI-ES
- A2C

**Features**:
- Hindsight Experience Replay Buffer (HER)
- Dynamic Movement Primitives (DMPs)
- Visual Inputs 

## Example
An example is given at the [main.py](main.py):
```python
import sys

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
agent = DDPG(sys.argv[1:], env, ReachTarget, obs_config)

# define number of training steps
training_episodes = 5000

# run agent
agent.run(training_episodes)
```
