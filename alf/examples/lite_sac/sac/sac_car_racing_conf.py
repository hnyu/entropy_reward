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

import torch

import gym
import numpy as np

import alf
from alf.algorithms.data_transformer import (
    RewardNormalizer, ObservationNormalizer, ImageScaleTransformer,
    FrameStacker)

from alf.environments.gym_wrappers import FrameGrayScale
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.algorithms.td_loss import TDLoss
from alf.algorithms.encoding_algorithm import EncodingAlgorithm

from alf.examples import sac_conf

DEBUG = False

if DEBUG:
    num_envs = 4
    initial_collect_steps = 1000
else:
    num_envs = 16
    initial_collect_steps = 50000

# environment config
alf.config(
    'create_environment',
    env_name="CarRacing-v0",
    num_parallel_environments=num_envs)

alf.config(
    "suite_gym.load",
    gym_env_wrappers=(FrameGrayScale, ),
    max_episode_steps=1000)

latent_size = 256
alf.config(
    "TrainerConfig",
    data_transformer_ctor=[
        partial(FrameStacker, stack_size=4),
        partial(ImageScaleTransformer, min=0., max=1.),
        partial(RewardNormalizer, clip_value=5.)
    ])

actor_network_cls = partial(
    ActorDistributionNetwork,
    input_preprocessors=alf.layers.Detach(),
    fc_layer_params=(latent_size, ) * 2,
    continuous_projection_net_ctor=partial(
        alf.networks.NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=alf.math.clipped_exp))
critic_network_cls = partial(
    CriticNetwork, joint_fc_layer_params=(latent_size, ) * 2)

alf.config(
    'SacAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    target_update_tau=0.005,
    #initial_log_alpha=math.log(0.1),
    critic_loss_ctor=TDLoss)

alf.config('calc_default_target_entropy', min_prob=0.1)

encoder_cls = partial(
    alf.networks.EncodingNetwork,
    conv_layer_params=((32, 8, 4), (64, 4, 2), (64, 3, 1)))

from alf.algorithms.encoding_algorithm import EncodingAlgorithm
alf.config('EncodingAlgorithm', encoder_cls=encoder_cls)

learning_rate = 3e-4
alf.config(
    'Agent',
    representation_learner_cls=EncodingAlgorithm,
    rl_algorithm_cls=SacAlgorithm,
    optimizer=alf.optimizers.AdamTF(lr=learning_rate))

# training config
alf.config(
    'TrainerConfig',
    temporally_independent_train_step=True,
    initial_collect_steps=initial_collect_steps,
    mini_batch_length=8,  # must use a large length
    unroll_length=5,
    mini_batch_size=256,
    num_updates_per_train_iter=1,
    num_iterations=0,
    num_env_steps=int(1e7),
    num_checkpoints=1,
    evaluate=True,
    num_evals=100,
    debug_summaries=True,
    summarize_grads_and_vars=False,
    summarize_first_interval=False,
    num_summaries=100,
    replay_buffer_length=100000)
