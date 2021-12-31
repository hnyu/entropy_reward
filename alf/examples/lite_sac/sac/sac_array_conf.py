# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import math

import alf
from alf.examples import sac_conf
from alf.examples.benchmarks.simple_control import simple_control_conf
from alf.examples.lite_sac.envs import simple_array


@alf.configurable
def conf(alpha_lr):
    pass


alf.config("create_environment", env_name='Array5-v0')

alf.config('suite_gym.load', max_episode_steps=50)

alf.config('Agent', optimizer=simple_control_conf.optimizer)

hidden_layers = (100, )

actor_distribution_network_cls = partial(
    alf.networks.ActorDistributionNetwork,
    fc_layer_params=hidden_layers,
    continuous_projection_net_ctor=partial(
        alf.networks.NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=alf.math.clipped_exp))

critic_network_cls = partial(
    alf.networks.CriticNetwork, joint_fc_layer_params=hidden_layers)

q_network_cls = partial(alf.networks.QNetwork, fc_layer_params=hidden_layers)

if alf.get_config_value("conf.alpha_lr") == "0":
    alpha_optimizer = alf.optimizers.AdamTF(lr=0)
else:
    alpha_optimizer = None

alf.config(
    'SacAlgorithm',
    actor_network_cls=actor_distribution_network_cls,
    critic_network_cls=critic_network_cls,
    q_network_cls=q_network_cls,
    initial_log_alpha=math.log(0.2),
    alpha_optimizer=alpha_optimizer,
    target_update_tau=0.005)

alf.config(
    "TrainerConfig",
    initial_collect_steps=5000,
    num_env_steps=50000,
    evaluate=True,
    data_transformer_ctor=None)

alf.config('calc_default_target_entropy', min_prob=0.184)
