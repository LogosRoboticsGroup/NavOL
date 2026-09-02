# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import numpy as np
import math
import torch
from typing import TYPE_CHECKING, Literal

from isaacsim.core.prims import XFormPrim
import isaacsim.core.utils.numpy.rotations as rot_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from navol.assets import DINGO_CAMERA_ROTS
from navol.training import CANONICAL_TRAINING
from .utils import usd_utils
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_asset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
    them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_object_state[env_ids].clone()

    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    asset.write_object_state_to_sim(root_states, env_ids=env_ids)


def reset_robot_with_cones(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, list],  # x, y, z, yaw
    asset_robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cone_red_cfg: SceneEntityCfg = SceneEntityCfg("cone_red"),
    asset_cone_green_cfg: SceneEntityCfg = SceneEntityCfg("cone_green"),
    asset_cone_blue_cfg: SceneEntityCfg = SceneEntityCfg("cone_blue"),
):
    asset_robot: RigidObject | Articulation = env.scene[asset_robot_cfg.name]
    asset_cone_red: RigidObject | Articulation = env.scene[asset_cone_red_cfg.name]
    asset_cone_green: RigidObject | Articulation = env.scene[asset_cone_green_cfg.name]
    asset_cone_blue: RigidObject | Articulation = env.scene[asset_cone_blue_cfg.name]
    # get default root state
    default_root_states = asset_robot.data.default_root_state[env_ids].clone()

    x_range_list = pose_range.get("x")
    y_range_list = pose_range.get("y")
    z_range_list = pose_range.get("z")
    yaw_range_list = pose_range.get("yaw")
    assert (
        len(x_range_list) == len(y_range_list) == len(z_range_list) == len(yaw_range_list)
    ), "The length of x, y, z, and yaw ranges should be the same."

    def tighten_range(range_list, tightness=0.2):
        tightened_ranges = []
        for start, end in range_list:
            new_start = start + tightness
            new_end = end - tightness
            if new_start < new_end:
                tightened_ranges.append((new_start, new_end))
            else:
                assert False, "Invalid range"
        return tightened_ranges

    x_range_list_robot = tighten_range(x_range_list)
    y_range_list_robot = tighten_range(y_range_list)
    z_range_list_robot = z_range_list
    yaw_range_list_robot = yaw_range_list

    random_robot_list = []
    random_cone_red_list = []
    random_cone_green_list = []
    random_cone_blue_list = []

    for i in range(len(env_ids)):
        chosen_indices = torch.randperm(len(x_range_list))[:4]
        for j in range(4):
            idx = chosen_indices[j]

            if j == 0:
                x_range = x_range_list_robot[idx]
                y_range = y_range_list_robot[idx]
                z_range = z_range_list_robot[idx]
                yaw_range = yaw_range_list_robot[idx]
            else:
                x_range = x_range_list[idx]
                y_range = y_range_list[idx]
                z_range = z_range_list[idx]
                yaw_range = yaw_range_list[idx]

            if j == 0:
                ranges = torch.tensor([x_range, y_range, z_range, yaw_range], device=asset_robot.device)
                rand_samples_robot = math_utils.sample_uniform(
                    ranges[:, 0], ranges[:, 1], (4), device=asset_robot.device
                )
                random_robot_list.append(rand_samples_robot)
            elif j == 1:
                ranges = torch.tensor([x_range, y_range, z_range], device=asset_robot.device)
                rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (3), device=asset_robot.device)
                random_cone_red_list.append(rand_samples)
            elif j == 2:
                ranges = torch.tensor([x_range, y_range, z_range], device=asset_robot.device)
                rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (3), device=asset_robot.device)
                random_cone_green_list.append(rand_samples)
            elif j == 3:
                ranges = torch.tensor([x_range, y_range, z_range], device=asset_robot.device)
                rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (3), device=asset_robot.device)
                random_cone_blue_list.append(rand_samples)

    robot_samples = torch.stack(random_robot_list, dim=0)  # (len(env_ids), 4)
    cone_red_samples = torch.stack(random_cone_red_list, dim=0)  # (len(env_ids), 3)
    cone_green_samples = torch.stack(random_cone_green_list, dim=0)  # (len(env_ids), 3)
    cone_blue_samples = torch.stack(random_cone_blue_list, dim=0)  # (len(env_ids), 3)

    rand_samples = torch.zeros((len(env_ids), 6), device=asset_robot.device)
    rand_samples[:, 0:3] = robot_samples[:, 0:3]
    rand_samples[:, 5] = robot_samples[:, 3]

    positions = default_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(default_root_states[:, 3:7], orientations_delta)
    velocities = default_root_states[:, 7:13]

    root_states = asset_cone_red.data.default_object_state[env_ids].clone()
    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    root_states[..., :3] += cone_red_samples.unsqueeze(1)
    asset_cone_red.write_object_state_to_sim(root_states, env_ids=env_ids)

    root_states = asset_cone_green.data.default_object_state[env_ids].clone()
    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    root_states[..., :3] += cone_green_samples.unsqueeze(1)
    asset_cone_green.write_object_state_to_sim(root_states, env_ids=env_ids)

    root_states = asset_cone_blue.data.default_object_state[env_ids].clone()
    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    root_states[..., :3] += cone_blue_samples.unsqueeze(1)
    asset_cone_blue.write_object_state_to_sim(root_states, env_ids=env_ids)


def reset_cones(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, list],  # x, y, z, yaw
    asset_robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cone_red_cfg: SceneEntityCfg = SceneEntityCfg("cone_red"),
    asset_cone_green_cfg: SceneEntityCfg = SceneEntityCfg("cone_green"),
    asset_cone_blue_cfg: SceneEntityCfg = SceneEntityCfg("cone_blue"),
):
    asset_robot: RigidObject | Articulation = env.scene[asset_robot_cfg.name]
    asset_cone_red: RigidObject | Articulation = env.scene[asset_cone_red_cfg.name]
    asset_cone_green: RigidObject | Articulation = env.scene[asset_cone_green_cfg.name]
    asset_cone_blue: RigidObject | Articulation = env.scene[asset_cone_blue_cfg.name]

    x_range_list = pose_range.get("x")
    y_range_list = pose_range.get("y")
    z_range_list = pose_range.get("z")
    yaw_range_list = pose_range.get("yaw")
    assert (
        len(x_range_list) == len(y_range_list) == len(z_range_list) == len(yaw_range_list)
    ), "The length of x, y, z, and yaw ranges should be the same."

    random_robot_list = []
    random_cone_red_list = []
    random_cone_green_list = []
    random_cone_blue_list = []

    for i in range(len(env_ids)):
        chosen_indices = torch.randperm(len(x_range_list))[:3]
        for j in range(3):
            idx = chosen_indices[j]

            x_range = x_range_list[idx]
            y_range = y_range_list[idx]
            z_range = z_range_list[idx]
            yaw_range = yaw_range_list[idx]

            if j == 0:
                ranges = torch.tensor([x_range, y_range, z_range], device=asset_robot.device)
                rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (3), device=asset_robot.device)
                random_cone_red_list.append(rand_samples)
            elif j == 1:
                ranges = torch.tensor([x_range, y_range, z_range], device=asset_robot.device)
                rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (3), device=asset_robot.device)
                random_cone_green_list.append(rand_samples)
            elif j == 2:
                ranges = torch.tensor([x_range, y_range, z_range], device=asset_robot.device)
                rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (3), device=asset_robot.device)
                random_cone_blue_list.append(rand_samples)

    cone_red_samples = torch.stack(random_cone_red_list, dim=0)  # (len(env_ids), 3)
    cone_green_samples = torch.stack(random_cone_green_list, dim=0)  # (len(env_ids), 3)
    cone_blue_samples = torch.stack(random_cone_blue_list, dim=0)  # (len(env_ids), 3)

    root_states = asset_cone_red.data.default_object_state[env_ids].clone()
    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    root_states[..., :3] += cone_red_samples.unsqueeze(1)
    asset_cone_red.write_object_state_to_sim(root_states, env_ids=env_ids)

    root_states = asset_cone_green.data.default_object_state[env_ids].clone()
    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    root_states[..., :3] += cone_green_samples.unsqueeze(1)
    asset_cone_green.write_object_state_to_sim(root_states, env_ids=env_ids)

    root_states = asset_cone_blue.data.default_object_state[env_ids].clone()
    root_states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    root_states[..., :3] += cone_blue_samples.unsqueeze(1)
    asset_cone_blue.write_object_state_to_sim(root_states, env_ids=env_ids)


reset_counter = 0
def pixelnav_reset(env: ManagerBasedEnv, 
                   env_ids: torch.Tensor, 
                   init_point_path:str,
                   height_offset:float,
                   robot_visible:bool,
                   light_enabled:bool,
                   robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    global reset_counter
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    np.random.seed(1234)
    sample_points = np.load(init_point_path)
    
    if not robot_visible:
        for i in range(env_ids.shape[0]):
            usd_utils.hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    if light_enabled:
        if reset_counter == 0:
            for light_idx,pts in enumerate(sample_points[:,0]):
                pts = pts + np.array([0.0, 0.0, 1.5])
                usd_utils.add_point_light(torch.as_tensor(pts, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
                                prim_path= f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}")
    
    random_robot_points = []
    random_goal_points = []
    random_init_orientions = []
    for i in range(env_ids.shape[0]):
        idx = int((i + reset_counter) % sample_points.shape[0])
        start_goal_pair = sample_points[idx]
        start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0])
        goal_points = np.array([start_goal_pair[2], start_goal_pair[3], 0])
        init_orientions = start_goal_pair[4]
        random_robot_points.append(start_points)
        random_goal_points.append(goal_points)
        random_init_orientions.append(init_orientions)
        
    random_robot_points = np.array(random_robot_points)
    random_goal_points = np.array(random_goal_points)
    random_init_orientions = np.array(random_init_orientions)
    random_init_orientions = torch.tensor(random_init_orientions, dtype=torch.float32, device=robot_asset.data.root_pos_w.device)
    tensor_robot_points = torch.tensor(random_robot_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
    tensor_goal_points = torch.tensor(random_goal_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_goal_points[:, 2] = tensor_goal_points[:, 2] + 1.5
    
    angle = random_init_orientions
    angle = angle.unsqueeze(-1).cpu().numpy()
    batch_init_rotation = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0, angle*0.0, angle), axis=-1))).to(robot_asset.data.root_pos_w.device)
    robot_asset.write_root_pose_to_sim(torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)),dim=-1),env_ids)
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_{env_id}/Goal", name="xform_view")
        goal_primview.set_world_poses(tensor_goal_points[i].unsqueeze(0),batch_init_rotation[i].unsqueeze(0))
    reset_counter += env_ids.shape[0]

def imagenav_reset(env: ManagerBasedEnv, 
                   env_ids: torch.Tensor, 
                   init_point_path:str,
                   height_offset:float,
                   camera_offset:float,
                   robot_visible:bool,
                   light_enabled:bool,
                   robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    global reset_counter
    np.random.seed(1234)
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    sample_points = np.load(init_point_path)
    
    if not robot_visible:
        for i in range(env_ids.shape[0]):
            usd_utils.hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    if light_enabled:
        if reset_counter == 0:
            for light_idx,pts in enumerate(sample_points[:,0]):
                pts = pts + np.array([0.0, 0.0, 1.5])
                add_point_light(torch.as_tensor(pts, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
                                prim_path= f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}")
    
    random_robot_points = []
    random_goal_points = []
    random_init_orientions = []
    for i in range(env_ids.shape[0]):
        idx = int((i + reset_counter) % sample_points.shape[0])
        start_goal_pair = sample_points[idx]
        start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0])
        goal_points = np.array([start_goal_pair[2], start_goal_pair[3], 0])
        init_orientions = start_goal_pair[4]
        random_robot_points.append(start_points)
        random_goal_points.append(goal_points)
        random_init_orientions.append(init_orientions)
        
    random_robot_points = np.array(random_robot_points)
    random_goal_points = np.array(random_goal_points)
    random_init_orientions = np.array(random_init_orientions)
    random_init_orientions = torch.tensor(random_init_orientions, dtype=torch.float32, device=robot_asset.data.root_pos_w.device)
    tensor_robot_points = torch.tensor(random_robot_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
    tensor_goal_points = torch.tensor(random_goal_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_goal_points[:, 2] = tensor_goal_points[:, 2] + 1.5
    
    angle = random_init_orientions
    angle = angle.unsqueeze(-1).cpu().numpy()
    batch_init_rotation = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0, angle*0.0, angle), axis=-1))).to(robot_asset.data.root_pos_w.device)
    robot_asset.write_root_pose_to_sim(torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)),dim=-1),env_ids)
    
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_{env_id}/goal_cam", name="xform_view")
        goal_image_point = tensor_goal_points[i]
        goal_image_point[2] = robot_asset.data.root_pos_w[i,2] + camera_offset
        goal_image_rot = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0 + np.pi/2, angle*0.0, angle - np.pi/2), axis=-1))).to(robot_asset.data.root_pos_w.device)
        goal_primview.set_world_poses(goal_image_point.unsqueeze(0),goal_image_rot)
        
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_{env_id}/Goal", name="xform_view")
        goal_primview.set_world_poses(tensor_goal_points[i].unsqueeze(0),batch_init_rotation[i].unsqueeze(0))
    reset_counter += env_ids.shape[0]

light_set = False
camera_set = False
def pointnav_reset(env: ManagerBasedEnv, 
                   env_ids: torch.Tensor, 
                   init_point_path:str,
                   height_offset:float,
                   robot_visible:bool,
                   light_enabled:bool,
                   sample_from_npy:bool = False,
                   rand_camera:bool = True,
                   height_range:tuple = CANONICAL_TRAINING.camera_height_range,
                   pitch_range:tuple = CANONICAL_TRAINING.camera_pitch_range,
                   robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                   camea_asset_cfg: SceneEntityCfg = SceneEntityCfg("camera_sensor")):
    global reset_counter
    global light_set
    global camera_set
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    camera_asset = env.scene[camea_asset_cfg.name]
    # np.random.seed(1234)
    device = robot_asset.data.root_pos_w.device
    
    # if light_enabled:
    #     if reset_counter == 0:
    #         for light_idx,pts in enumerate(sample_points):
    #             pts = pts + np.array([0.0, 0.0, 1.5])
    #             usd_utils.add_point_light(torch.as_tensor(pts, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
    #                             prim_path= f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}")
    
    if not robot_visible:
        for i in range(env_ids.shape[0]):
            usd_utils.hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    
    if not camera_set and rand_camera:
        camera_height = (torch.rand(env_ids.shape[0], device=device) * (height_range[1] - height_range[0]) + height_range[0])
        print(f"Sampled camera heights: {camera_height}")
        # camera_height = torch.full((env_ids.shape[0], ), 1.0, device=device)
        
    if hasattr(env, '_recent_positions_z'):
        env._recent_positions_z[env_ids] = -99999.
    action_manager = env.unwrapped.action_manager.get_term("joint_combined")
    if sample_from_npy:
        sample_points = np.load(init_point_path)
        random_robot_points = []
        random_goal_points = []
        random_init_orientions = []
        # reset_counter = 0
        for i in range(env_ids.shape[0]):
            idx = int((i + reset_counter) % sample_points.shape[0])
            # idx = np.random.randint(0, sample_points.shape[0])
            # print(f"Env: {int(env_ids[i])} Sampled index: {idx}")
            # idx = 0
            start_goal_pair = sample_points[idx]
            # start_goal_pair = np.array([-9.68074989, 10.08109283, -3.6512301,  -1.21869898, -1.04158175, -3.6512301, -0.96163287], dtype=np.float32)
            start_points = np.array([start_goal_pair[0], start_goal_pair[1], start_goal_pair[2]]) * action_manager.cfg.scene_scale
            goal_points = np.array([start_goal_pair[3], start_goal_pair[4], start_goal_pair[5]]) * action_manager.cfg.scene_scale
            init_orientions = start_goal_pair[6]
            # if reset_counter % 2 == 0:
            # init_orientions = start_goal_pair[6] + math.pi
            random_robot_points.append(start_points)
            random_goal_points.append(goal_points)
            random_init_orientions.append(init_orientions)
            
        random_robot_points = np.array(random_robot_points)
        random_goal_points = np.array(random_goal_points)
        random_init_orientions = np.array(random_init_orientions)
    else:
        if not camera_set and rand_camera:
            start_goal_pairs = action_manager.generate_goal(env_ids, height=camera_height)
        else:
            start_goal_pairs = action_manager.generate_goal(env_ids)
        random_robot_points = start_goal_pairs[:, :3]
        random_goal_points = start_goal_pairs[:, 3:6]
        random_init_orientions = start_goal_pairs[:, 6] + np.random.rand(start_goal_pairs.shape[0]) * np.pi * 2 / 3 - np.pi / 3
        # random_init_orientions = np.random.rand(start_goal_pairs.shape[0]) * 2 * np.pi
       
    tensor_robot_points = torch.tensor(random_robot_points, dtype=torch.float32, device=device) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
    tensor_goal_points = torch.tensor(random_goal_points, dtype=torch.float32, device=device) + env.scene.env_origins[env_ids]
    
    angle = random_init_orientions[..., None]
    batch_init_rotation = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0, angle*0.0, angle), axis=-1)), device=device)
    robot_asset.write_root_pose_to_sim(torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)),dim=-1),env_ids)
    
    if not camera_set and rand_camera:
        camera_pos = torch.zeros(env_ids.shape[0], 3, device=device)
        camera_pos[:, 2] = camera_height
        camera_rot = torch.tensor(DINGO_CAMERA_ROTS, device=device).reshape(1, 4).repeat(env_ids.shape[0], 1)
        zeros = torch.zeros(env_ids.shape[0], device=device)
        pitchs = (torch.rand(env_ids.shape[0], device=device) * (pitch_range[1] - pitch_range[0]) + pitch_range[0]) * torch.pi / 180.0
        camera_rot_random = math_utils.quat_from_euler_xyz(zeros, pitchs, zeros)                            # (N,4), wxyz
        camera_rot = math_utils.quat_mul(camera_rot_random, camera_rot)
        camera_asset.set_world_poses(camera_pos, camera_rot, env_ids=env_ids)
        camera_set = True
    
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_{env_id}/Goal", name="xform_view")
        goal_primview.set_world_poses(tensor_goal_points[i].unsqueeze(0),batch_init_rotation[i].unsqueeze(0))
        # goal_primview.set_visibilities([True])
    reset_counter += env_ids.shape[0]
    return

    ###################

    tensor_robot_points =torch.tensor([-9.271987, 12.584974, -3.621037], dtype=torch.float32, device=device) + env.scene.env_origins[env_ids]

    # tensor_robot_points =torch.tensor([4.70544, -3.92243, -1.37476], dtype=torch.float32, device=device) + env.scene.env_origins[env_ids]
    # tensor_goal_points = torch.tensor([0.280454, -0.736033, -1.37476], dtype=torch.float32, device=device) + env.scene.env_origins[env_ids]
    
    # new scene
    tensor_goal_points = torch.tensor([-0.39095178, 1.3786833, -3.621037], dtype=torch.float32, device=device) + env.scene.env_origins[env_ids]
    
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    # tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset +0.2
    # tensor_robot_points[:, 2] = 0.0 + height_offset
    # tensor_goal_points[:, 2] = tensor_goal_points[:, 2] + 1.5
    
    angle = np.array([2.8337874], dtype=np.float32)[None].repeat(env_ids.shape[0], 0)
    angle = np.array([-0.7419472], dtype=np.float32)[None].repeat(env_ids.shape[0], 0)
    batch_init_rotation = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0, angle*0.0, angle), axis=-1))).to(device)
    robot_asset.write_root_pose_to_sim(torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)),dim=-1),env_ids)
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_{env_id}/Goal", name="xform_view")
        goal_primview.set_world_poses(tensor_goal_points[i].unsqueeze(0),batch_init_rotation[i].unsqueeze(0))

        # scene = XFormPrim(prim_paths_expr=f"/World/Scene", name="xform_scene")


        # if not light_set:
        #     usd_utils.add_point_light(torch.as_tensor(tensor_robot_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
        #                             prim_path= f"/World/envs/env_{env_ids[0]}/point_light_{0}")
        #     usd_utils.add_point_light(torch.as_tensor(tensor_goal_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
        #                             prim_path= f"/World/envs/env_{env_ids[0]}/point_light_{1}")
        #     light_set = True
                                
    # for i in range(env_ids.shape[0]):
    #     usd_utils.hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    return

    
def pointnav_eval_reset(env: ManagerBasedEnv, 
                   env_ids: torch.Tensor, 
                   init_point_path:str,
                   height_offset:float,
                   robot_visible:bool,
                   light_enabled:bool,
                   robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),):
    global reset_counter
    global light_set
    global camera_set
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    # np.random.seed(1234)
    device = robot_asset.data.root_pos_w.device

    if not robot_visible:
        for i in range(env_ids.shape[0]):
            usd_utils.hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    
    if hasattr(env, '_recent_positions_z'):
        env._recent_positions_z[env_ids] = -99999.
    sample_points = np.load(init_point_path)
    
    if light_enabled:
        if reset_counter == 0:
            for light_idx,pts in enumerate(sample_points[:, :3]):
                pts = pts + np.array([0.0, 0.0, 1.5])
                usd_utils.add_point_light(torch.as_tensor(pts, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
                                prim_path= f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}")
    
    random_robot_points = []
    random_goal_points = []
    random_init_orientions = []
    action_manager = env.unwrapped.action_manager.get_term("joint_combined")
    for i in range(env_ids.shape[0]):
        idx = int((i + reset_counter) % sample_points.shape[0])
        print(f"Env: {int(env_ids[i])} Sampled index: {idx}")
        # idx = np.random.randint(0, sample_points.shape[0])
        # print(f"Env: {int(env_ids[i])} Sampled index: {idx}")
        # idx = 0
        start_goal_pair = sample_points[idx]
        # start_goal_pair = np.array([-9.68074989, 10.08109283, -3.6512301,  -1.21869898, -1.04158175, -3.6512301, -0.96163287], dtype=np.float32)
        if len(start_goal_pair) == 7:
            start_points = np.array([start_goal_pair[0], start_goal_pair[1], start_goal_pair[2]]) * action_manager.cfg.scene_scale
            goal_points = np.array([start_goal_pair[3], start_goal_pair[4], start_goal_pair[5]]) * action_manager.cfg.scene_scale
            init_orientions = start_goal_pair[6]
        elif len(start_goal_pair) == 5:
            start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0.0]) * action_manager.cfg.scene_scale
            goal_points = np.array([start_goal_pair[2], start_goal_pair[3], 0.0]) * action_manager.cfg.scene_scale
            init_orientions = start_goal_pair[4]
        else:
            raise ValueError("Invalid start_goal_pair length")
        # if reset_counter % 2 == 0:
        # init_orientions = start_goal_pair[6] + math.pi
        random_robot_points.append(start_points)
        random_goal_points.append(goal_points)
        random_init_orientions.append(init_orientions)
        
    random_robot_points = np.array(random_robot_points)
    random_goal_points = np.array(random_goal_points)
    random_init_orientions = np.array(random_init_orientions)
       
    tensor_robot_points = torch.tensor(random_robot_points, dtype=torch.float32, device=device) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
    tensor_goal_points = torch.tensor(random_goal_points, dtype=torch.float32, device=device) + env.scene.env_origins[env_ids]
    
    angle = random_init_orientions[..., None]
    batch_init_rotation = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0, angle*0.0, angle), axis=-1)), device=device)
    robot_asset.write_root_pose_to_sim(torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)),dim=-1),env_ids)
    
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_{env_id}/Goal", name="xform_view")
        goal_primview.set_world_poses(tensor_goal_points[i].unsqueeze(0),batch_init_rotation[i].unsqueeze(0))
        # goal_primview.set_visibilities([True])
    reset_counter += env_ids.shape[0]
    return

def exploration_reset(env: ManagerBasedEnv, 
                      env_ids: torch.Tensor, 
                      init_point_path:str,
                      height_offset:float,
                      robot_visible:bool,
                      light_enabled:bool,
                      robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    global reset_counter
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    np.random.seed(1234)
    sample_points = np.load(init_point_path)
    
    if not robot_visible:
        for i in range(env_ids.shape[0]):
            usd_utils.hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    if light_enabled:
        if reset_counter == 0:
            for light_idx,pts in enumerate(sample_points[:,0]):
                pts = pts + np.array([0.0, 0.0, 1.5])
                usd_utils.add_point_light(torch.as_tensor(pts, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
                                prim_path= f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}")
                
    random_robot_points = []
    random_init_orientions = []
    for i in range(env_ids.shape[0]):
        idx = int((i + reset_counter) % sample_points.shape[0])
        start_goal_pair = sample_points[idx]
        start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0])
        init_orientions = start_goal_pair[4]
        random_robot_points.append(start_points)
        random_init_orientions.append(init_orientions)
        
    random_robot_points = np.array(random_robot_points)
    tensor_robot_points = torch.tensor(random_robot_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    random_init_orientions = np.array(random_init_orientions)
    random_init_orientions = torch.tensor(random_init_orientions, dtype=torch.float32, device=robot_asset.data.root_pos_w.device)
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
        
    angle = random_init_orientions
    angle = angle.unsqueeze(-1).cpu().numpy()
    batch_init_rotation = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0, angle*0.0, angle), axis=-1))).to(robot_asset.data.root_pos_w.device)
    robot_asset.write_root_pose_to_sim(torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)),dim=-1),env_ids)
    reset_counter += env_ids.shape[0]
