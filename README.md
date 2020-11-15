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

#### Optional
If you want to use the C++ implementation of prioritized experience replay, you need to build it first.\
Go to 'agents/ddpg_backend/sum_tree_cpp' and run the build script, which downloads all dependencies and compiles the library:
```shell script
cd agents/ddpg_backend/sum_tree_cpp
sudo chmod u+x build.sh
./build.sh 
```
This will create a the library `sum_tree_cpp.cpython-37m-x86_64-linux-gnu.so`, which you need to either install at\
a location, which is already in your `PYTHONPATH`, or to add it manually to your `PYTHONPATH`:
```shell script
export PYTHONPATH="${PYTHONPATH}:/path/to/your/lib.so"
```
If you do so, consider adding the latter to your `.bashrc` file, if you want the path to be added at startup.  


## Currently included Agents and Features
**Agents**:
- Deep Deterministic Policy Gradient (DDPG)
- Twin Delayed DDPG (TD3)
- OpenAI-ES

**Features**:
- Prioritized Experience Replay (PER)
    - Implemented in Python and C++ (approx. 10-15% performance gain compared to Python implementation)

### In Progress:
- Including a ROS-API

## Planned Agents and Features
**Agents**:
- PI² 
- CMA-ES
- A2C

**Features**:
- Hindsight Experience Replay Buffer (HER)
- Dynamic Movement Primitives (DMPs)
- Visual Inputs 

## Example
An example is given at the [main.py](main.py):
```python
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
```
Note that you can add the path to your own configuration-file. 
```
python main.py --config /path/to/your/config-file.yaml
```
Therefore, copy and modify the default
[configuration-file](agents/config/default_config.yaml).
