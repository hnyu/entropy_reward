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
import gym

import alf
from alf.examples import sac_conf
from alf.examples.benchmarks.locomotion import locomotion_conf
from alf.algorithms.data_transformer import RewardScaling


@alf.configurable
def conf(alpha_lr=None):
    pass


class WorstRewardInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['worst_reward'] = 0.
        return obs, reward, done, info


alf.config('suite_gym.load', gym_env_wrappers=(WorstRewardInfoWrapper, ))

alf.config('Agent', optimizer=locomotion_conf.optimizer)

if alf.get_config_value("conf.alpha_lr") == "0":
    alpha_optimizer = alf.optimizers.AdamTF(lr=0)
else:
    alpha_optimizer = None

alf.config(
    'SacAlgorithm',
    actor_network_cls=locomotion_conf.actor_distribution_network_cls,
    critic_network_cls=locomotion_conf.critic_network_cls,
    #initial_log_alpha=math.log(0.1),
    alpha_optimizer=alpha_optimizer,
    reproduce_locomotion=True,
    target_update_tau=0.005)

alf.config('calc_default_target_entropy', min_prob=0.184)
