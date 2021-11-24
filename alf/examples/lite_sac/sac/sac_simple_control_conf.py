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
import math

import alf
from alf.examples import sac_conf
from alf.examples.benchmarks.simple_control import simple_control_conf

alf.config('Agent', optimizer=simple_control_conf.optimizer)

alf.config(
    'SacAlgorithm',
    actor_network_cls=simple_control_conf.actor_distribution_network_cls,
    critic_network_cls=simple_control_conf.critic_network_cls,
    #initial_log_alpha=math.log(0.1),
    target_update_tau=0.005)

alf.config('calc_default_target_entropy', min_prob=0.1)
