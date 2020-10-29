"""
Run DQN on grid world.
"""
import gym
import torch
from torch import nn as nn
import argparse

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.networks import Mlp
from rlkit.torch.conv_networks import CNN
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from env_brainbow.env_synthetic import EnvBrainbow


def experiment(variant):
    fov, delta, num_ch = 33, 8, 3
    expl_env = EnvBrainbow('0:data/training_sample/training_sample_1.tif'
                           ',1:data/training_sample/training_sample_2.tif',
                           coord_interval=4, img_mean=128, img_stddev=33, num_ch=num_ch, fov=fov, delta=delta, seed=0)
    eval_env = EnvBrainbow('0:data/training_sample/training_sample_1.tif'
                           ',1:data/training_sample/training_sample_2.tif',
                           coord_interval=4, img_mean=128, img_stddev=33, num_ch=num_ch, fov=fov, delta=delta, seed=0)
    obs_dim = expl_env.observation_space.low.shape  # 33, 33, 3
    action_dim = eval_env.action_space.n  # 2
    kernel_sizes = [4, 4, 3]
    n_channels = [32, 64, 64]
    strides = [2, 2, 1]
    paddings = [0, 0, 0]
    hidden_sizes = [512]

    qf = CNN(
        input_width=fov,
        input_height=fov,
        input_channels=num_ch,
        output_size=action_dim,
        kernel_sizes=kernel_sizes,
        n_channels=n_channels,
        strides=strides,
        paddings=paddings,
        hidden_sizes=hidden_sizes,
        batch_norm_conv=True,
        batch_norm_fc=False
    )
    target_qf = CNN(
        input_width=fov,
        input_height=fov,
        input_channels=num_ch,
        output_size=action_dim,
        kernel_sizes=kernel_sizes,
        n_channels=n_channels,
        strides=strides,
        paddings=paddings,
        hidden_sizes=hidden_sizes,
        batch_norm_conv=True,
        batch_norm_fc=False
    )

    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space),
        eval_policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant['trainer_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='DQN-runs')
    parser.add_argument("--env", type=str, default='EnvBrainbow')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    variant = dict(
        algorithm="DQN",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=3E-4,
        ),
    )

    setup_logger(exp_prefix='dqn-' + args.env,
                 variant=variant,
                 text_log_file="debug.log",
                 variant_log_file="variant.json",
                 tabular_log_file="progress.csv",
                 snapshot_mode="gap_and_last",
                 snapshot_gap=50,
                 log_tabular_only=False,
                 log_dir=None,
                 git_infos=None,
                 script_name=None,
                 # **create_log_dir_kwargs
                 base_log_dir='./data',
                 exp_id=0,
                 seed=0)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)