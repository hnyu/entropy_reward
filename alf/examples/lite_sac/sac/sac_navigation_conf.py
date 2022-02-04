# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import alf
from alf.examples import sac_conf
from alf.examples.benchmarks.simple_control import simple_control_conf
from alf.examples.lite_sac.envs import navigation

alf.config("create_environment", env_name='ToyNavigation-v0')

alf.config('suite_gym.load', max_episode_steps=50)

alf.config('Agent', optimizer=simple_control_conf.optimizer)

alf.config(
    'SacAlgorithm',
    actor_network_cls=simple_control_conf.actor_distribution_network_cls,
    critic_network_cls=simple_control_conf.critic_network_cls,
    target_update_tau=0.005)

alf.config("TrainerConfig", num_env_steps=int(2e5))

alf.config('calc_default_target_entropy', min_prob=0.2)
