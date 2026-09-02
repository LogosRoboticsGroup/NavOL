# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import numpy as np
from collections import deque
import torch
from typing import TYPE_CHECKING

from isaacsim.core.prims import XFormPrim
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def arrival_terminal_check(env: ManagerBasedRLEnv,
                           robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_threshold: float = 1.0, velocity_threshold: float = 0.5):
    robot_asset = env.scene[robot_asset_cfg.name]
    robot_pos = robot_asset.data.root_pos_w
    goal_primview = XFormPrim(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view")
    goal_pos = goal_primview.get_world_poses()[0]
    robot_vel = robot_asset.data.root_lin_vel_w
    distance = torch.square(robot_pos[:,0:2] - goal_pos[:,0:2]).sum(axis=1).sqrt()
    velocity = torch.abs(robot_vel).sum(axis=1)

    arrive = (distance < distance_threshold) & (velocity < velocity_threshold)
    arrived_indices = torch.nonzero(arrive, as_tuple=True)[0]
    if len(arrived_indices) > 0:
        print(f"\033[1;36mEnv {arrived_indices.tolist()} Arrived!!!!\033[0m")
    return arrive

def stuck_terminal_check(env: ManagerBasedRLEnv,
                         robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                         window_size: int = 5,
                         threshold: float = 0.05):
    if not hasattr(env, '_recent_positions'):
        env._recent_positions = deque(maxlen=window_size)
    robot_asset = env.scene[robot_asset_cfg.name]
    pos = robot_asset.data.root_pos_w[:, :2].cpu().numpy()  # 只看x, y
    env._recent_positions.append(pos)
    if len(env._recent_positions) < window_size:
        return False 
    current = torch.tensor(env._recent_positions[-1], device=env.device)
    # max_dist = np.max(np.linalg.norm(current - np.array(p)) for p in list(env._recent_positions)[:-1], axis=1)
    recent_positions = np.array(list(env._recent_positions)[:-1])
    recent_positions = torch.tensor(recent_positions, device=env.device)
    distances = torch.norm(current - recent_positions, dim=-1)
    max_dist = torch.max(distances, dim=0).values
    # if bool(max_dist < threshold):
    #     print(" Stuck!!!!")
    #     return True
    # return False
    stuck = max_dist < threshold
    stuck_indices = torch.where(stuck)[0]
    if len(stuck_indices) > 0:
        try:
            action_manager = env.unwrapped.action_manager.get_term("joint_combined")
            scene_id = action_manager.scene_id
            print(f"\033[1;36mEnv {stuck_indices.tolist()} Stuck!!!!\033[0m, sim_id: {scene_id[stuck_indices].tolist()}")
            
        except:
            print(f"\033[1;36mEnv {stuck_indices.tolist()} Stuck!!!!\033[0m")
            # print("No joint_combined action manager found.")
    return stuck

def fall_check(env: ManagerBasedRLEnv,
                         robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                         window_size: int = 5,
                         threshold: float = 1):
    robot_asset = env.scene[robot_asset_cfg.name]
    pos = robot_asset.data.root_pos_w[:, 2]
    num_envs = pos.shape[0]
    
    if not hasattr(env, '_recent_positions_z'):
        env._recent_positions_z = torch.full((num_envs, ), -99999., device=env.device)
        
    fall = env._recent_positions_z - pos > threshold
    env._recent_positions_z = torch.maximum(env._recent_positions_z, pos)
    
    try:
        action_manager = env.unwrapped.action_manager.get_term("joint_combined")
        is_valids = action_manager.is_valids
        if (~is_valids).sum() > 0:
            print("is_valids:", is_valids)
        fall = fall | (~is_valids)
    except:
        pass
        # print("No joint_combined action manager found.")
        
    fall_indices = torch.where(fall)[0]
    if len(fall_indices) > 0:
        print(f"\033[1;36mEnv {fall_indices.tolist()} Fall!!!!\033[0m")
    return fall
        
    for env_id in range(num_envs):
        if env_id not in env._recent_positions_z:
            env._recent_positions_z[env_id] = deque(maxlen=window_size)
        env._recent_positions_z[env_id].append(pos[env_id])
    
    fall = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    
    for env_id in range(num_envs):
        if len(env._recent_positions_z[env_id]) < window_size:
            continue
            
        current_pos = pos[env_id]
        recent_positions = np.array(list(env._recent_positions_z[env_id])[:-1])
        distances = np.abs(current_pos - recent_positions)
        max_dist = np.max(distances)
        if max_dist > threshold:
            fall[env_id] = True
            env._recent_positions_z[env_id].clear()

    fall_indices = torch.where(fall)[0]
    if len(fall_indices) > 0:
        print(f"\033[1;36mEnv {fall_indices.tolist()} Fall!!!!\033[0m")
    return fall

def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")
