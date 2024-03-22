# Overcooked! - Develop diverse cooking tasks for multi-agent cooperation – Variant 3

For KCL BSc Computer Science (Artificial Intelligence) Dissertation

Contents:

- [Abstract](#0.Abstract)
- [Installation](#1.installation)
- [Bayesian Delegation](#BayesianDelegation)
- [DQN](#DQN)
- [Environments and Recipes](docs/environments.md)
- [Design and Customization](docs/design.md)

## 0.Abstract

The Gifs are directly taken from the original "Too Many Cook!" Github repository, which can be accessed here: https://github.com/rosewang2008/gym-cooking

<p align="center">
    <img src="images/2_open_salad.gif" width=260></img>
    <img src="images/2_partial_tl.gif" width=260></img>
    <img src="images/2_full_salad.gif" width=260></img>
</p>

Multi-Agent Reinforcement Learning (MARL) is a topic long-discussed in machine learning research, which covers various applications ranging from healthcare systems to automated logistics design. With a publication "Too Many Cooks!" \cite{mitOvercooked} carried out by MIT and Harvard students that discusses agent coordination in partially observable situations, this project expands upon the concept of "Role Differentiation", attempting to observe the effect of different algorithms on role allocation and its impact on the original algorithm for subtask achievements.

This project presents the specific implementation and evaluation of the role allocation algorithm on expanded action sets, covering the comparison between the original simulation, role-optimized solution, and other proposed solutions. Subsequently, the project explores the usage of Deep-Q Learning on multiple agents to compare performances and achieve a different way of agent training. To increase difficulty in agent coordination, the project also explored the possibility of resource scarcity and various agent execution algorithms in subtask selections.

## 1.Installation

### 1.1 Create Environment

Please try with pip3 / python3 if the following does not work on the machine.

```
conda create -n OvercookedV3 python==3.7.16
conda activate OvercookedV3
python -m pip install -e.
cd gym_cooking
```

To download the tensorflow for deep learning:

```
python -m pip install tensorflow
```

## Bayesian Delegation

The command structure has the following arguments:

```
python main.py --num-agents <number> --level <level name> --model1 <model name> --model2 <model name> --model3 <model name> --model4 <model name> --role <role name> --record
```

In the original design, the agents were able to choose to run the different models, including:

- `bd` to run Bayesian Delegation,
- `up` for Uniform Priors,
- `dc` for Divide & Conquer,
- `fb` for Fixed Beliefs, and
- `greedy` for Greedy.

However, with the implementation of role differentiation, please run the structure with `bd` only to avoid system issues.

The other parameters include `<number>` for the number of agents, `<level>` for specifying the level that is played, and `<role>`, which can be selected from the following:

- `extreme` for single agents taking all the work,
- `optimal` for the optimal role allocation,
- `unbalanced` for slightly unbalanced work segregation,
- `three` for optimal role separation design,
- `none` for the original non-role specified simulation.

## Usage

Here, we discuss how to run a single experiment, run our code in manual mode, and re-produce results in our paper. For information on customizing environments, observation/action spaces, and other details, please refer to our section on [Design and Customization](docs/design.md)

For the code below, make sure that you are in **gym-cooking/gym_cooking/**. This means, you should be able to see the file `main.py` in your current directory.

<p align="center">
    <img src="images/2_open.png" width=260></img>
    <img src="images/3_partial.png" width=260></img>
    <img src="images/4_full.png" width=260></img>
</p>

### Running an experiment

The basic structure of our commands is the following:

`python main.py --num-agents <number> --level <level name> --model1 <model name> --model2 <model name> --model3 <model name> --model4 <model name>`

where `<number>` is the number of agents interacting in the environment (we handle up to 4 agents), `level name` are the names of levels available under the directory `cooking/utils/levels`, omitting the `.txt`.

The `<model name>` are the names of models described in the paper. Specifically `<model name>` can be replaced with:

- `bd` to run Bayesian Delegation,
- `up` for Uniform Priors,
- `dc` for Divide & Conquer,
- `fb` for Fixed Beliefs, and
- `greedy` for Greedy.

For example, running the salad recipe on the partial divider with 2 agents using Bayesian Delegation looks like:
`python main.py --num-agents 2 --level partial-divider_salad --model1 bd --model2 bd`

Or, running the tomato-lettuce recipe on the full divider with 3 agents, one using UP, one with D&C, and the third with Bayesian Delegation:
`python main.py --num-agents 2 --level full-divider_tl --model1 up --model2 dc --model3 bd`

Although our work uses object-oriented representations for observations/states, the `OvercookedEnvironment.step` function returns _image observations_ in the `info` object. They can be retrieved with `info['image_obs']`.

### Additional commands

The above commands can also be appended with the following flags:

- `--record` will save the observation at each time step as an image in `misc/game/record`.

### Manual control

To manually control agents and explore the environment, append the `--play` flag to the above commands. Specifying the model names isn't necessary but the level and the number of agents is still required. For instance, to manually control 2 agents with the salad task on the open divider, run:

`python main.py --num-agents 2 --level open-divider_salad --play`

This will open up the environment in Pygame. Only one agent can be controlled at a time -- the current active agent can be moved with the arrow keys and toggled by pressing `1`, `2`, `3`, or `4` (up until the actual number of agents of course). Hit the Enter key to save a timestamped image of the current screen to `misc/game/screenshots`.

### Reproducing paper results

To run our full suite of computational experiments (self-play and ad-hoc), we've provided the scrip `run_experiments.sh` that runs our experiments on 20 seeds with `2` agents.

To run on `3` agents, modify `run_experiments.sh` with `num_agents=3`.

### Creating visualizations

To produce the graphs from our paper, navigate to the `gym_cooking/misc/metrics` directory, i.e.

1. `cd gym_cooking/misc/metrics`.

To generate the timestep and completion graphs, run:

2. `python make_graphs.py --legend --time-step`
3. `python make_graphs.py --legend --completion`

This should generate the results figures that can be found in our paper.

Results for homogenous teams (self-play experiments):
![graphs](images/graphs.png)

Results for heterogeneous teams (ad-hoc experiments):
![heatmaps](images/heatmaps.png)
