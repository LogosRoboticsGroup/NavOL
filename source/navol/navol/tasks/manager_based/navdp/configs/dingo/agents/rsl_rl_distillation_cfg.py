# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from navol.wrapper.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlDistillationNavdpCfg, RslRlDistillationAlgorithmCfg
from navol.paths import model_path
from navol.training import CANONICAL_TRAINING
# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class DingoPointgoalRunnerCfg(RslRlOnPolicyRunnerCfg):
    device = "cuda:0"
    num_steps_per_env = CANONICAL_TRAINING.rollout_steps
    max_iterations = CANONICAL_TRAINING.iterations
    save_interval = 100
    experiment_name = "dingo_pointgoal_distillation"
    empirical_normalization = False
    resume = False
    # resume = True
    # load_run = "2025-01-11_20-23-09"
    policy = RslRlDistillationNavdpCfg(
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
        ft_denoising_steps=2,
        use_critic=True,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="DiffusionDistillation",
        num_learning_epochs=CANONICAL_TRAINING.learning_epochs,
        num_mini_batches=CANONICAL_TRAINING.mini_batches,
        gradient_length=1,
        learning_rate=CANONICAL_TRAINING.learning_rate,
        max_grad_norm=0.01,
        lambda_critic=CANONICAL_TRAINING.lambda_critic,
    )
