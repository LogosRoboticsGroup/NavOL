# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def reach_goal(
    env: ManagerBasedRLEnv, threshold, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    pos_r = asset.data.root_pos_w - env.scene.env_origins
    quat_r = asset.data.root_quat_w

    pos_goal = env.observation_manager._obs_buffer['policy']['goal_w']

    reward = (torch.norm(pos_r - pos_goal, dim=1) < threshold).float()

    return reward

class goal_dis(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        self.dis = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        # root_env = asset.data.root_pos_w - env.scene.env_origins
        pos_r = asset.data.root_pos_w - env.scene.env_origins
        print("pos_ro: ", pos_r)
        quat_r = asset.data.root_quat_w
        pos_goal = env.observation_manager._obs_buffer['policy']['goal_w']
        dis = torch.norm(pos_r - pos_goal, dim=1)
        start_id = torch.where(self.dis == 0.0)
        reward = self.dis - dis
        reward[start_id] = 0.0
        self.dis = dis
        return reward * (dis > 0.25)

    def reset(self, env_ids: torch.Tensor):
        self.dis[env_ids] = 0.0

class goal_dis_z(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        self.dis = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        root_env = asset.data.root_pos_w - env.scene.env_origins
        pos_goal = env.observation_manager._obs_buffer['policy']['goal_w']
        dis = torch.abs(root_env[:, 2] - pos_goal[:, 2])
        start_id = torch.where(self.dis == 0.0)
        reward = self.dis - dis
        reward[start_id] = 0.0
        self.dis = dis
        return reward

    def reset(self, env_ids: torch.Tensor):
        self.dis[env_ids] = 0.0

def goal_heading_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    root_env = asset.data.root_pos_w - env.scene.env_origins
    pos_goal = env.observation_manager._obs_buffer['policy']['goal_pose']

    root_quat = asset.data.root_quat_w

    # Compute direction to goal in yaw
    direction_to_goal = pos_goal[:, :2] - root_env[:, :2]
    goal_yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])

    # Convert quaternion to yaw
    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    current_yaw = torch.atan2(t3, t4)

    # Calculate yaw error
    yaw_error = torch.abs(goal_yaw - current_yaw)
    yaw_error = torch.where(yaw_error > torch.pi, 2 * torch.pi - yaw_error, yaw_error)

    # Convert yaw error to linear reward
    linear_reward = 2 * yaw_error / torch.pi
    dis = torch.norm(root_env - pos_goal, dim=1)

    return 1 - linear_reward * (dis > 0.25)

def stand_still_at_goal(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    pos_r = asset.data.root_pos_w - env.scene.env_origins
    quat_r = asset.data.root_quat_w
    pos_goal = env.observation_manager._obs_buffer['policy']['goal_pose']

    dis = torch.norm(pos_r - pos_goal, dim=1)
    return (dis < 0.25) * (1 / (torch.norm(env.action_manager.get_term("joint_vel").joint_velocities, dim=1) + 0.4))

def track_lin_vel_xy_exp_command(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(
            env.action_manager.get_term("joint_vel").joint_velocities[:, :2] - asset.data.root_lin_vel_b[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp_command(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(
        env.action_manager.get_term("joint_vel").joint_velocities[:, 2] - asset.data.root_ang_vel_b[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)

class track_predict_trajectory_command(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)

    def __call__(
        self, env: ManagerBasedRLEnv, std, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        policy_actions = env.action_manager.get_term("joint_combined").policy_actions
        command_actions = env.action_manager.get_term("joint_combined").command_actions
        error = torch.square(policy_actions - command_actions).sum(dim=-1).sum(dim=-1)
        return torch.exp(-error / std**2)

class track_predict_velocity_command(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)

    def __call__(
        self, env: ManagerBasedRLEnv, action, action_nav, std, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        
        asset = env.scene[asset_cfg.name]
        velocity_command_nav = env.action_manager.get_term("joint_combined").cmd_velocity_controller
        velocity_command = env.action_manager.get_term("joint_combined").velocity_controller
        error = torch.square(velocity_command - velocity_command_nav).sum(dim=-1).sum(dim=-1)
        return torch.exp(-error / std**2)
        # 线性速度:余弦相似度
        nav_lin_norm = torch.norm(velocity_command_nav[:, :2], dim=1)
        cmd_lin_norm = torch.norm(velocity_command[:, :2], dim=1)
        
        valid_mask = (nav_lin_norm > 1e-6) & (cmd_lin_norm > 1e-6)
        lin_rew = torch.zeros_like(nav_lin_norm)
        
        if torch.any(valid_mask):
            cosine_sim = torch.sum(
                velocity_command[valid_mask, :2] * velocity_command_nav[valid_mask, :2], dim=1
            ) / (nav_lin_norm[valid_mask] * cmd_lin_norm[valid_mask])
            magnitude_match = torch.exp(-torch.abs(nav_lin_norm[valid_mask] - cmd_lin_norm[valid_mask]))
            
            lin_rew[valid_mask] = cosine_sim * magnitude_match

        # 角速度奖励
        yaw_error = torch.square(velocity_command_nav[:, 2] - velocity_command[:, 2])
        ang_rew = torch.exp(-yaw_error / std**2)

        angular_component = torch.abs(velocity_command_nav[:, 2])
        turn_factor = torch.sigmoid(angular_component * 5.0 - 2.0)
        forward_factor = 1.0 - turn_factor * 0.7
        
        # print("yaw_error: ", yaw_error)
        # print("nav: ",nav_lin_norm)
        # print("velocity: ", cmd_lin_norm)

        return lin_rew * forward_factor + ang_rew