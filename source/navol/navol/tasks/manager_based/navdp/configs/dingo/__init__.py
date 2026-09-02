# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, navdp_imagegoal_cfg, navdp_pixelgoal_cfg, navdp_pointgoal_cfg, navdp_nogoal_cfg

##
# Register Gym environments for NavDP tasks (Evaluation only, no training).
# NavDP uses pre-trained models served via HTTP API for trajectory planning.
##

gym.register(
    id="nogoal",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # Note: ManagerBasedEnv, not RL version
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navdp_nogoal_cfg.DingoExplorationCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DingoNogoalRunnerCfg",
    },
)

gym.register(
    id="pointgoal_train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navdp_pointgoal_cfg.DingoPointNavCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DingoPointgoalRunnerCfg",
    },
)

gym.register(
    id="pointgoal_train_distillation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navdp_pointgoal_cfg.DingoPointNavCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DingoPointgoalRunnerCfg",
    },
)

gym.register(
    id="pointgoal_eval_distillation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navdp_pointgoal_cfg.DingoPointNavCfg_Eval,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DingoPointgoalRunnerCfg",
    },
)

gym.register(
    id="pointgoal",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navdp_pointgoal_cfg.DingoPointNavCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DingoPointgoalRunnerCfg",
    },
)

gym.register(
    id="imagegoal",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navdp_imagegoal_cfg.DingoImageNavCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DingoImagegoalRunnerCfg",
    },
)

gym.register(
    id="pixelgoal",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navdp_pixelgoal_cfg.DingoPixelNavCfg,
    },
)
