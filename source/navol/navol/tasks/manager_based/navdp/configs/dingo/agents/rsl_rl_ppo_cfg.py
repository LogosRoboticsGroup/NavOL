# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from navol.wrapper.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoNavdpCfg
from navol.paths import model_path
# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class DingoPointgoalRunnerCfg(RslRlOnPolicyRunnerCfg):
    device = "cuda:0"
    num_steps_per_env = 140
    max_iterations = 8000
    save_interval = 100
    experiment_name = "dingo_pointgoal"
    empirical_normalization = False
    resume = False
    # resume = True
    # load_run = "2025-01-11_20-23-09"
    policy = RslRlPpoNavdpCfg(
        class_name="NavdpPolicy",
        pretrained_model_path=str(model_path("navdp-cross-modal.ckpt", must_exist=False)),
        task_name="pointgoal",
        memory_size=8,
        image_size=224,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        channels=3,
        stop_threshold=-2.0,
        ft_denoising_steps=10
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="DiffusionPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1,
    )

@configclass
class DingoImagegoalRunnerCfg(RslRlOnPolicyRunnerCfg):
    device = "cuda:0"
    num_steps_per_env = 40
    max_iterations = 8000
    save_interval = 100
    experiment_name = "dingo_imagegoal"
    empirical_normalization = False
    resume = False
    # resume = True
    # load_run = "2025-01-11_20-23-09"
    policy = RslRlPpoNavdpCfg(
        class_name="NavdpPolicy",
        pretrained_model_path=str(model_path("navdp-cross-modal.ckpt", must_exist=False)),
        task_name="imagegoal"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1,
    )


@configclass
class DingoNogoalRunnerCfg(RslRlOnPolicyRunnerCfg):
    device = "cuda:0"
    num_steps_per_env = 40
    max_iterations = 8000
    save_interval = 100
    experiment_name = "dingo_nogoal"
    empirical_normalization = False
    resume = False
    # resume = True
    # load_run = "2025-01-11_20-23-09"
    load_checkpoint = str(model_path("navdp-cross-modal.ckpt", must_exist=False))
    policy = RslRlPpoNavdpCfg(
        class_name="NavdpPolicy",
        pretrained_model_path=str(model_path("navdp-cross-modal.ckpt", must_exist=False)),
        task_name="nogoal"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1,
    )

