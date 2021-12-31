# Do you need the entropy reward (in practice)?

This repo releases the experiment code for the technical report

> Do You Need the Entropy Reward (in Practice)? Yu et al., arxiv?? 2022.

The code is a forked version of [ALF](https://github.com/HorizonRobotics/alf). For detailed information and installation of ALF, please refer to its [readme](https://github.com/HorizonRobotics/alf/blob/pytorch/README.md) and we will skip it here.

## What is this technical report about?

We identify a side effect of the entropy reward, called "*reward inflation*", in episodic MDPs. This effect could obscure or completely change the original MDP.
We also conduct a large-scale empirical study of SAC as a representative algorithm, showing that the properties of good exploration, training convergence and stability, and policy robustness of MaxEnt RL, might result *more from entropy regularizing policy improvement than from entropy regularizing policy evaluation*. In most cases, the entropy reward is unnecessary as an intrinsic reward.

## Experiments

Below we show how to launch the training scripts of different experiments in the report. All scripts will be run under ``<ALF_ROOT>/alf/examples/lite_sac``, where ``<ALF_ROOT>`` is your ALF root
directory. For each specific command, we use the placeholder token ``<METHOD>`` to represent one of the three comparison methods:

| METHOD     | method name in the report |
|------------|---------------------------|
| sac        | SAC                       |
| lite_sac   | SACLite                   |
| sac_plus   | SACZero                   |

To run the same training command for a different method, you just need to replace ``<METHOD>`` with the desired value (e.g., lite_sac) in
the first column of the table above.

We will also use the placeholder ``<TRAIN_DIR>`` to represent the training directory selected. This directory is a Tensorboard log directory and
you can use ``tensorboard --logdir=<TRAIN_DIR>`` to view the training curves.

### Episodic SimpleChain

#### Fixed alpha

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_array_conf.py --conf_param="conf.alpha_lr=0"
```

#### Tunable alpha

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_array_conf.py --conf_param="conf.alpha_lr=None"
```

### Infinite-horizon SimpleChain

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_array_conf.py --conf_param="conf.alpha_lr=None" --conf_param="create_environment.env_name='InfiniteArray5-v0'"
```

### MoJoCo manipulation

(This experiment requires installing [MuJoCo](https://github.com/openai/mujoco-py) first.)

#### Initial alpha=1

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_fetch_conf.py --conf_param="create_environment.env_name='<TASK_NAME>-v1'"
```

where ``<TASK_NAME>`` is ``FetchPush``, ``FetchReach``, ``FetchSlide``, or ``FetchPickAndPlace``.

#### Initial alpha=0.1

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_fetch_conf.py --conf_param="create_environment.env_name='<TASK_NAME>-v1'" --conf_param="SacAlgorithm.initial_log_alpha=-2.3026"
```

### Box2D

(This experiment requires installing [Box2D](https://pypi.org/project/Box2D/) first.)

#### BipedalWalker with initial alpha=1

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_bipedalwalker_conf.py --conf_param="create_environment.env_name='<TASK_NAME>-v2'"
```

where ``<TASK_NAME>`` is either ``BipedalWalker`` or ``BipedalWalkerHardcore``.


#### BipedalWalker with initial alpha=0.1

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_bipedalwalker_conf.py --conf_param="create_environment.env_name='<TASK_NAME>-v2'" --conf_param="SacAlgorithm.initial_log_alpha=-2.3026"
```

#### LunarLanderContinuous with initial alpha=1

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_simple_control_conf.py --conf_param="create_environment.env_name='LunarLanderContinuous-v2'"
```

#### LunarLanderContinuous with initial alpha=0.1

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_simple_control_conf.py --conf_param="create_environment.env_name='LunarLanderContinuous-v2'" --conf_param="SacAlgorithm.initial_log_alpha=-2.3026"
```

#### CarRacing with initial alpha=1

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_car_racing_conf.py
```

#### CarRacing with initial alpha=0.1

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_car_racing_conf.py --conf_param="SacAlgorithm.initial_log_alpha=-2.3026"
```

### Locomotion

#### MuJoCo

(This experiment requires installing [MuJoCo](https://github.com/openai/mujoco-py) first.)

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_locomotion_conf.py --conf_param="create_environment.env_name='<TASK_NAME>-v3'"
```

where ``<TASK_NAME>`` is ``Hopper``, ``Walker2d``, ``Ant``, or ``HalfCheetah``.

#### DM control suite

(This experiment requires installing [DM control suite](https://github.com/deepmind/dm_control) first.)

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_dmc_conf.py --conf_param="create_environment.env_name='<TASK_NAME>'"
```

where ``<TASK_NAME>`` is ``pendulum:swingup``, ``fish:upright``, ``fish:swim``, ``swimmer:swimmer6``, ``ball_in_cup:catch``, ``reacher:hard``,
``finger:spin``, or ``finger:turn_hard``.

### Multi-objective RL

(This experiment requires installing our customized [Safety Gym](https://github.com/hnyu/safety-gym) first.)

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_point_conf.py --conf_param="create_environment.env_name='<TASK_NAME>-v0'"
```

where ``<TASK_NAME>`` is ``Safexp-PointGoal1``, ``Safexp-PointGoal2``, ``Safexp-PointPush1``, ``Safexp-PointPush2``, ``Safexp-PointButton1``, or ``Safexp-PointButton2``.

### Robustness

#### Dynamics robustness

Training without the obstacle:

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_navigation_conf.py
```

Evaluation with the obstacle:

``` python
python -m alf.bin.play --root_dir <TRAIN_DIR> --num_episodes 500 --norender --conf_param="create_environment.env_name='ToyNavigationObstacle-v0'"
```

#### Reward robustness

``` python
python -m alf.bin.train --root_dir <TRAIN_DIR> --conf <METHOD>/<METHOD>_locomotion_conf.py --conf_param="conf.alpha_lr=0" --conf_param="SacAlgorithm.initial_log_alpha=-2.3026" --conf_param="create_environment.env_name='<TASK_NAME>-v3'"
```

where ``<TASK_NAME>`` is ``Hopper``, ``Walker2d``, ``Ant``, or ``HalfCheetah``. The "worst-case" reward curve is shown in the tab "Metrics_vs_EnvironmentSteps/worst_reward".

## Citation
If you find this work useful in your research, please consider citing

```
 @article{Yu2022Entropy,
    author={Haonan Yu and Haichao Zhang and Wei Xu},
    title={Do you need the entropy reward (in practice)?},
    jounral={arXiv},
    year={2022}
 }
```
