from __future__ import annotations

import os
import math
import cv2
import trimesh
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict
from scipy.spatial.transform import Rotation as R

import isaaclab.utils.string as string_utils
import omni.log
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import JointVelocityAction

from isaacsim.core.prims import XFormPrim
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
import habitat_sim
from habitat.utils.visualizations import maps
from .utils.path_smoothing import get_smooth_points

from . import actions_cfg
from .utils.tracking_utils import MPC_Controller, DifferentialController, MPC_Controller_Batch, TorchPIDController_Batch, MPC_Controller_simple
from .utils.navmesh_utils import init_navmesh, navmesh_find_path, navmesh_generate_goal, compute_navmesh
from navol.training import compute_critic_targets, select_policy_action


def velocity_from_local_command(cfg: actions_cfg.VelocityCommandActionCfg, vel_xy):
    vx = vel_xy[..., 0]
    vy = vel_xy[..., 1]

    # === 2. 计算目标方向角（相对车头 x 轴） ===
    heading_angle = torch.atan2(vy, vx)  # (-pi, pi]

    # === 3. 计算前进线速度 ===
    v_lin = torch.norm(vel_xy, dim=-1) * cfg.vel_scale

    # 若目标在后方（x<0），则取负号表示倒车
    back_mask = vx < 0
    v_lin[back_mask] *= -1.0

    # === 4. 根据偏角抑制线速度 ===
    # 这里用 cos(angle) 但限定在 ±pi/2 以内，
    # 若超过 ±90°（完全背向），线速度逐渐趋近 0。
    # angle_factor = torch.cos(torch.clamp(2 * torch.abs(heading_angle), 0, torch.pi / 2))
    # angle_factor = (torch.abs(torch.cos(heading_angle)) ** (1.0 + cfg.factor_scale_vel))
    # angle_factor = torch.cos(torch.clamp(torch.abs(heading_angle), 0, torch.pi / 2))
    # angle_factor = 2 * torch.cos(heading_angle).abs()
    angle_factor_vel = torch.cos(heading_angle.clamp(0, torch.pi / 2)) ** (1.0 + cfg.factor_scale_vel)
    angle_factor_ang = 1 / torch.clamp_min(torch.cos(heading_angle.clamp(0, torch.pi / 2)) ** (1.0 + cfg.factor_scale_ang), 0.5)
    # angle_factor_vel = (torch.abs(torch.cos(heading_angle)) ** (1.0 + cfg.factor_scale_vel))
    # angle_factor_ang = (torch.abs(torch.sin(heading_angle)) ** (1.0 + cfg.factor_scale_ang))
    # angle_factor_ang = 1 / torch.clamp_min(angle_factor_vel, 0.5)
    
    # angle_factor_vel = torch.clamp(angle_factor_vel, 0.0, 1.0)
    v_lin = v_lin * angle_factor_vel
    v_lin = v_lin.clamp(-cfg.max_linear_speed, cfg.max_linear_speed).unsqueeze(-1)

    # === 5. 旋转速度由横向分量决定 ===
    v_ang = heading_angle * cfg.angular_scale
    v_ang = v_ang * angle_factor_ang
    v_ang = v_ang.clamp(-cfg.max_angular_speed, cfg.max_angular_speed).unsqueeze(-1)

    # === 6. 拼接最终指令 ===
    velocity_command = torch.cat([vel_xy, v_ang], dim=-1)
    velocity = torch.cat([v_lin, v_ang], dim=-1)
    # print("velocity_from_local_command: ", vel_xy.tolist(), velocity.tolist())
    return velocity_command, velocity
    
    
    
    
    heading_angle = torch.atan2(vel_xy[..., 1], vel_xy[..., 0])
    vel_w = (heading_angle).clamp(-cfg.max_angular_speed, cfg.max_angular_speed)[:, None]

    angle_threshold = torch.pi / 6
    angle_mask = torch.abs(heading_angle) > angle_threshold
    angle_factor = torch.cos(torch.clamp(torch.abs(heading_angle), 0, torch.pi/2))
    angle_factor = torch.clamp(angle_factor, 0.5, 1.0).unsqueeze(-1)

    if angle_mask.any():
        vel_xy[angle_mask] *= angle_factor[angle_mask]
        vel_w[angle_mask] /= angle_factor[angle_mask]
    
    velocity_command = torch.cat([vel_xy, vel_w], dim=-1)

    vel_norm = torch.norm(vel_xy, dim=-1, keepdim=True) * cfg.vel_scale
    vel_norm.clamp_max_(cfg.max_linear_speed)

    velocity = torch.cat([vel_norm, vel_w], dim=-1)
    # velocity = torch.cat([vel_norm, vel_w], dim=-1) * 4
    return velocity_command, velocity

class NavDPMPCAction(JointVelocityAction):
    """Action term for NavDP navigation with MPC control."""
    
    cfg: actions_cfg.NavDPMPCActionCfg

    def __init__(self, cfg: actions_cfg.NavDPMPCActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # 初始化 MPC 控制器
        self.mpc_controllers = [None] * self.num_envs
        self.differential_controller = DifferentialController(
            name="navdp_control",
            wheel_radius=cfg.wheel_radius,
            wheel_base=cfg.wheel_base
        )
        
        self._get_camera_info()


    def process_actions2(self, actions: torch.Tensor):
        """处理观察并生成导航动作"""
        with torch.inference_mode():
            self._get_camera_info()
            # 转换到世界坐标系
            actions = actions.reshape(self.num_envs, self.cfg.action_steps, 3)
            actions = torch.cumsum(actions / 4.0, dim=1)

            camera_pos = self._env.scene.sensors['camera_sensor'].data.pos_w
            camera_rot = math_utils.matrix_from_quat(self._env.scene.sensors['camera_sensor'].data.quat_w_world)
            points_local = torch.cat([actions[..., :2], torch.zeros_like(actions[..., :1])], dim=-1)
            trajectory_world2 = (camera_pos[:, None] + points_local @ camera_rot.transpose(-1, -2))[..., :2]
            # mpc = MPC_Controller_Batch(
            #     trajectory_world2, 
            #     desired_v=self.cfg.speed,
            #     v_max=self.cfg.max_linear_speed,
            #     w_max=self.cfg.max_angular_speed
            # )
            mpc = TorchPIDController_Batch(
                trajectory_world2, 
                desired_v=self.cfg.speed,
                v_max=self.cfg.max_linear_speed,
                w_max=self.cfg.max_angular_speed
            )
            x0 = torch.stack([camera_pos[:,0], camera_pos[:,1], torch.arctan2(camera_rot[:,1,0], camera_rot[:,0,0])], dim=-1)
            # control_commands = mpc.solve(x0)[:, 1]
            control_commands = mpc.solve(x0)
            joint_velocities = self.differential_controller.forward_torch(control_commands)
            self.joint_velocities = joint_velocities
                
    def process_actions(self, actions: torch.Tensor):
        """处理观察并生成导航动作"""
        actions = actions.reshape(self.num_envs, self.cfg.action_steps, 3)
        self.actions = actions
        trajectory = torch.cumsum(actions / 4.0, dim=1)
        self.trajectory_xy = trajectory[..., :2]
        vel_xy = self.trajectory_xy[:, 1] - self.trajectory_xy[:, 0]
        self.velocity_command, self.velocity_controller = velocity_from_local_command(self.cfg, vel_xy)
        self.joint_velocities = self.differential_controller.forward_torch(self.velocity_controller)

    def apply_actions(self):
        if self.cfg.apply_actions:
            self._asset.set_joint_velocity_target(self.joint_velocities, joint_ids=self._joint_ids)
        
    def _mpc_control(self, trajectory_world):
        """MPC生成速度命令"""
        robot_vel = self._env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[:, :2].norm(dim=-1).cpu().numpy()
        robot_ang_vel = self._env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[:, 2].cpu().numpy()
        x0 = np.stack([self.camera_pos[:,0], self.camera_pos[:,1], np.arctan2(self.camera_rot[:,1,0], self.camera_rot[:,0,0]), robot_vel, robot_ang_vel],axis=-1)
        
        control_commands = []
        
        for i in range(self.num_envs):
            opt_u_controls, _ = self.mpc_controllers[i].solve(x0[i,:3])
            v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
            control_commands.append([v, w])
        
        return torch.tensor(control_commands, device=self.device)

    def _get_camera_info(self):
        self.camera_pos = self._env.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
        camera_rot_quat = self._env.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
        camera_rot_quat = camera_rot_quat[:,[1, 2, 3, 0]]
        self.camera_rot = R.from_quat(camera_rot_quat).as_matrix()
        # return camera_pos, camera_rot

    def get_task_command(self):
        """获取当前任务命令"""
        pass

    @property
    def action_dim(self) -> int:
        return self.cfg.action_steps * 3  # diffusion predict_size
    
class VelocityCommandAction(JointVelocityAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    """The configuration of the action term."""
    cfg: actions_cfg.VelocityCommandActionCfg
    
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.differential_controller = DifferentialController(
            name="navdp_control",
            wheel_radius=cfg.wheel_radius,
            wheel_base=cfg.wheel_base
        )
        
        self.sim = init_navmesh(cfg.mesh_path)
        self.path_points = None
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_.*/Goal", name="xform_view")
    
    def process_actions(self, actions: torch.Tensor):
        
        robot_pos = self._env.scene["robot"].data.root_pos_w - self._env.scene.env_origins
        robot_pos_nav = robot_pos[:, [1, 2, 0]].cpu().numpy()

        robot_quat = self._env.scene["robot"].data.root_quat_w
        robot_quat_inv = math_utils.quat_inv(robot_quat)
        
        goal_pos = self.goal_primview.get_world_poses()[0]
        goal_pos_nav = goal_pos[:, [1, 2, 0]].cpu().numpy()

        path_points, factor = navmesh_find_path(self.sim, robot_pos_nav, goal_pos_nav, sample_num=self.cfg.predict_size)
        
        factor = torch.tensor(factor, device=self.device, dtype=torch.float32)
        self.path_points = torch.tensor(path_points, device=self.device, dtype=torch.float32)
        robot_quat_inv = robot_quat_inv.unsqueeze(1).expand(-1, self.path_points.shape[1], -1)
        self.path_points_local = math_utils.quat_apply(robot_quat_inv, self.path_points - robot_pos.unsqueeze(1))[..., :2]

        vel_navmesh_xy = self.path_points_local[:, 1] - self.path_points_local[:, 0]

        self.velocity_command, self.velocity_controller = velocity_from_local_command(self.cfg, vel_navmesh_xy)
        self.joint_velocities = self.differential_controller.forward_torch(self.velocity_controller)

        trajectory = torch.cat([self.path_points_local, torch.zeros_like(self.path_points_local[..., :1])], dim=-1)
        actions = (trajectory - torch.cat([torch.zeros_like(trajectory[:, :1]), trajectory[:, :-1]],dim=1)) * 4.0
        self.actions = actions
        
        # if self.velocity_command_val.min() < 0:
        #     print("angle_factor: ", angle_factor)

        # print("********** command velocity ***********: ", self.velocity_command_val)
        # print(f'\033[4;33m速度大小: {velocity}\033[0m')
        # print(f'\033[5;33m指引角速度: {velocity_command_w}\033[0m\n')


        # self.velocity_command_nav = torch.where(
        #     heading_angle > 3.14/180 * 10,
        #     torch.cat([torch.zeros_like(velocity_command_xy), velocity_command_w], dim=-1),
        #     torch.cat([velocity_command_xy, velocity_command_w], dim=-1),
        # )

        # self.velocity_command = torch.cat([velocity_command_xy, velocity_command_w], dim=-1)
        # TODO: 有的点离得太近导致速度过小

    def apply_actions(self):
        if self.cfg.apply_actions:
            self._asset.set_joint_velocity_target(self.joint_velocities, joint_ids=self._joint_ids)

    @property
    def action_dim(self) -> int:
        return 0
    
    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized or self.path_points is None:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 1.5
        # # -- resolve the scales and quaternions
        vel_navmesh = (self.path_points[:, 1] - self.path_points[:, 0])
        heading_angle = torch.atan2(vel_navmesh[..., 1], vel_navmesh[...,0])
        zeros = torch.zeros_like(heading_angle)
        vel_des_arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        
        vel_des_arrow_scale, _ = self._resolve_xy_velocity_to_arrow(torch.nn.functional.normalize(self.velocity_command[:, :2], dim=-1))
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(torch.nn.functional.normalize(self.robot.data.root_lin_vel_b[:, :2], dim=-1))
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

class VelocityCombinedAction(JointVelocityAction):
    """Joint action term that combines MPC control and velocity command without sub-actions."""

    cfg: actions_cfg.VelocityCombinedActionCfg
    
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        self.differential_controller = DifferentialController(
            name="navdp_control",
            wheel_radius=cfg.wheel_radius,
            wheel_base=cfg.wheel_base
        )
        
        self.sim = []
        for idx, mesh_path in enumerate(cfg.mesh_paths):
            try:
                self.sim.append(init_navmesh(mesh_path, radius=cfg.navmesh_radius))
            except Exception as e:
                raise ValueError(f"Failed to initialize navmesh for scene {idx} with path {mesh_path}: {e}")

        self.path_points = None
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_.*/Goal", name="xform_view")
        
        self.mpc_actions = None
        self.command_actions = None
        self.mpc_joint_velocities = None
        self.command_joint_velocities = None
        self.velocity_controller = None
        self.mpc_controllers = [None] * self.num_envs
        
        self.use_policy = torch.tensor([True] * self.num_envs, device=self.device)
        self.start_point = [None for _ in range(self.num_envs)]
        self.end_point = [None for _ in range(self.num_envs)]
        self.scene_id = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.get_action_type()
        self.spacing = 10.0
    
    def get_action_type(self, env_ids=None):
        if env_ids is None:
            env_ids = range(self.num_envs)
        for env_id in env_ids:
            use_policy = select_policy_action(
                self.cfg.action_type,
                self.cfg.action_rand_p,
                float(np.random.rand()),
            )
            self.use_policy[env_id] = use_policy
    
    def process_actions(self, actions: torch.Tensor):
        self._process_command_actions(actions)
        self._process_policy_actions(actions)


        if self.cfg.use_mpc:
            # actions_reshaped = actions.reshape(self.num_envs, self.cfg.action_steps, 3)
            # actions = torch.cumsum(actions_reshaped / 4.0, dim=1)
            # points_local = torch.cat([actions[..., :2], torch.zeros_like(actions[..., :1])], dim=-1)
            
            path_points_local = torch.where(
                self.use_policy[:, None, None],
                self.trajectory_xy,
                self.path_points_local
            )
            
            points_local = torch.cat([path_points_local, torch.zeros_like(path_points_local[..., :1])], dim=-1)
            camera_pos = self._env.scene.sensors['camera_sensor'].data.pos_w
            camera_rot = math_utils.matrix_from_quat(self._env.scene.sensors['camera_sensor'].data.quat_w_world)
            trajectory_world = (camera_pos[:, None] + points_local @ camera_rot.transpose(-1, -2))[..., :2]
            
            x0 = torch.stack([camera_pos[:,0], camera_pos[:,1], torch.arctan2(camera_rot[:,1,0], camera_rot[:,0,0])], dim=-1)
            control_commands = []
            for idx in range(trajectory_world.shape[0]):
                if self.mpc_controllers[idx] is None:
                    self.mpc_controllers[idx] = MPC_Controller(
                        desired_v=self.cfg.speed,
                        v_max=self.cfg.max_linear_speed,
                        w_max=self.cfg.max_angular_speed
                    )
                opt_u_controls, opt_x_states = self.mpc_controllers[idx].solve(x0[idx,:3].cpu().numpy(), trajectory_world[idx].cpu().numpy())
                v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
                control_command = torch.tensor([v, w], device=actions.device)
                control_commands.append(control_command)
            control_commands = torch.stack(control_commands, dim=0)
                
            self.mpc_joint_velocities = self.differential_controller.forward_torch(control_commands)
            self.joint_velocities = self.mpc_joint_velocities
        else:
            self.joint_velocities = torch.where(
                self.use_policy[:, None],
                self.policy_joint_velocities,
                self.command_joint_velocities
            )

    def _process_policy_actions(self, actions: torch.Tensor):
        """ from NavDPMPCAction """
        actions_reshaped = actions.reshape(self.num_envs, self.cfg.action_steps, 3)
        self.policy_actions = actions_reshaped
        trajectory = torch.cumsum(actions_reshaped / 4.0, dim=1)
        self.trajectory_xy = trajectory[..., :2]
        vel_xy = self.trajectory_xy[:, 1] - self.trajectory_xy[:, 0]
        
        velocity_command, self.policy_velocity_controller = velocity_from_local_command(self.cfg, vel_xy)
        self.policy_velocity_controller[~self.is_valids] = 0.0
        self.policy_joint_velocities = self.differential_controller.forward_torch(self.policy_velocity_controller)
    
    def _process_command_actions(self, actions: torch.Tensor):
        """ from VelocityCommandAction """
        robot_pos = self._env.scene["robot"].data.root_pos_w - self._env.scene.env_origins
        robot_pos_nav = self.isaacsim_to_navmesh_coords(robot_pos)

        robot_quat = self._env.scene["robot"].data.root_quat_w
        robot_quat_inv = math_utils.quat_inv(robot_quat)
        
        goal_pos = self.goal_primview.get_world_poses()[0]
        goal_pos_nav = self.isaacsim_to_navmesh_coords(goal_pos)
        
        path_points_list = []
        distances_list = []
        is_valid_list = []
        for i, idx in enumerate(self.scene_id.cpu().numpy()):
            if idx >= len(self.sim):
                path_point_nav = robot_pos_nav[i][None].repeat(self.cfg.predict_size, axis=0)
                distance = np.zeros((self.cfg.predict_size,))
                print(f"Warning: Scene ID {idx} exceeds the number of provided navmeshes {len(self.sim)}")
                # raise ValueError(f"Scene ID {idx} exceeds the number of provided navmeshes {len(self.sim)}")
                is_valid = False
            else:
                path_point_nav, distance, is_valid = navmesh_find_path(self.sim[idx], robot_pos_nav[i], goal_pos_nav[i], sample_num=self.cfg.predict_size, search_radius=self.cfg.search_radius, idx=idx, start_point=self.start_point[i], end_point=self.end_point[i])
            
            path_points_list.append(path_point_nav)
            distances_list.append(distance)
            is_valid_list.append(is_valid)
        path_points_nav = np.stack(path_points_list, axis=0)
        distances = np.stack(distances_list, axis=0)
        is_valids = np.stack(is_valid_list, axis=0)
        
        self.path_points = self.navmesh_to_isaacsim_coords(path_points_nav)
        robot_quat_inv = robot_quat_inv.unsqueeze(1).expand(-1, self.path_points.shape[1], -1)
        self.path_points_local = math_utils.quat_apply(robot_quat_inv, self.path_points - robot_pos.unsqueeze(1))[..., :2]
        self.is_valids = torch.tensor(is_valids, device=self.device, dtype=torch.bool)
        self.distances = torch.tensor(distances, device=self.device, dtype=torch.float32)
        self.critic_values = compute_critic_targets(
            self.distances,
            safe_distance=self.cfg.critic_safe_distance,
            progress_alpha=self.cfg.critic_progress_alpha,
            collision_reduction=self.cfg.critic_collision_reduction,
        )
        # self.critic_values = 10.0 * (self.distances[:, 1] - self.distances[:, 0])
        # breakpoint()
        vel_navmesh_xy = self.path_points_local[:, 1] - self.path_points_local[:, 0]

        velocity_command, self.command_velocity_controller = velocity_from_local_command(self.cfg, vel_navmesh_xy)
        self.command_joint_velocities = self.differential_controller.forward_torch(self.command_velocity_controller)
        trajectory = torch.cat([self.path_points_local, torch.zeros_like(self.path_points_local[..., :1])], dim=-1)
        command_actions = (trajectory - torch.cat([torch.zeros_like(trajectory[:, :1]), trajectory[:, :-1]], dim=1)) * 4.0
        
        self.command_actions = command_actions

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self.joint_velocities, joint_ids=self._joint_ids)
    
    def generate_goal(self, env_ids, height=None):
        samples = []
        for i in range(env_ids.shape[0]):
            sim_id = np.random.randint(0, len(self.sim))
            # sim_id = 1
            print("Generating goal for env_id: ", env_ids[i].item(), " using sim_id: ", sim_id)
            if height is not None:
                compute_navmesh(self.sim[sim_id], height=height[i].item(), radius=self.cfg.navmesh_radius)
            result = navmesh_generate_goal(self.sim[sim_id], sim_id=sim_id)
            sample = result['sample']
            sample[[2, 5]] += sim_id * self.spacing
            sample[:6] *= self.cfg.scene_scale
            samples.append(sample)
            self.start_point[env_ids[i]] = result['start_point_nav']
            self.end_point[env_ids[i]] = result['end_point_nav']
            if self.scene_id is None:
                self.scene_id = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.scene_id[env_ids[i]] = sim_id
        start_goal_pairs = np.array(samples)
        self.get_action_type(env_ids=env_ids)
        return start_goal_pairs
    
    def isaacsim_to_navmesh_coords(self, points):
        points = points.cpu().numpy() / self.cfg.scene_scale
        dims = len(points.shape) - 1
        scene_id = self.scene_id.cpu().numpy()
        for _ in range(dims - 1):
            scene_id = scene_id[..., None]
        points[..., 2] -= scene_id * self.spacing
        points[..., 1] *= -1
        points_nav = points[..., [0, 2, 1]].copy()
        return points_nav
    
    def navmesh_to_isaacsim_coords(self, points_nav):
        dims = len(points_nav.shape) - 1
        points = points_nav[..., [0, 2, 1]].copy()
        points[..., 1] *= -1
        scene_id = self.scene_id.cpu().numpy()
        for _ in range(dims - 1):
            scene_id = scene_id[..., None]
        points[..., 2] += scene_id * self.spacing
        points = torch.tensor(points, device=self.device, dtype=torch.float32) * self.cfg.scene_scale
        return points
    
    @property
    def action_dim(self) -> int:
        return self.cfg.action_steps * 3
    

class VelocityActionEval(JointVelocityAction):
    """Joint action term that combines MPC control and velocity command without sub-actions."""

    cfg: actions_cfg.VelocityCombinedActionCfg
    
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        self.differential_controller = DifferentialController(
            name="navdp_control",
            wheel_radius=cfg.wheel_radius,
            wheel_base=cfg.wheel_base
        )
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.is_valids = None
        self.critic_values = None
        self.mpc_controllers = [None] * self.num_envs
        
        print("VelocityActionEval initialized.")
        
    def process_actions(self, actions: torch.Tensor):
        self._process_policy_actions(actions)
        if self.cfg.use_mpc:
            path_points_local = self.trajectory_xy
            
            points_local = torch.cat([path_points_local, torch.zeros_like(path_points_local[..., :1])], dim=-1)
            camera_pos = self._env.scene.sensors['camera_sensor'].data.pos_w
            camera_rot = math_utils.matrix_from_quat(self._env.scene.sensors['camera_sensor'].data.quat_w_world)
            trajectory_world = (camera_pos[:, None] + points_local @ camera_rot.transpose(-1, -2))[..., :2]
            
            x0 = torch.stack([camera_pos[:,0], camera_pos[:,1], torch.arctan2(camera_rot[:,1,0], camera_rot[:,0,0])], dim=-1)
            control_commands = []
            for idx in range(trajectory_world.shape[0]):
                if self.mpc_controllers[idx] is None:
                    self.mpc_controllers[idx] = MPC_Controller(
                        desired_v=self.cfg.speed,
                        v_max=self.cfg.max_linear_speed,
                        w_max=self.cfg.max_angular_speed,
                    )
                opt_u_controls, opt_x_states = self.mpc_controllers[idx].solve(x0[idx,:3].cpu().numpy(), trajectory_world[idx].cpu().numpy())
                v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
                control_command = torch.tensor([v, w], device=actions.device)
                control_commands.append(control_command)
            control_commands = torch.stack(control_commands, dim=0)
                
            self.mpc_joint_velocities = self.differential_controller.forward_torch(control_commands)
            self.joint_velocities = self.mpc_joint_velocities
        else:
            self.joint_velocities = self.policy_joint_velocities
            
    def _process_policy_actions(self, actions: torch.Tensor):
        """ from NavDPMPCAction """
        actions_reshaped = actions.reshape(self.num_envs, self.cfg.action_steps, 3)
        self.policy_actions = actions_reshaped
        trajectory = torch.cumsum(actions_reshaped / 4.0, dim=1)
        self.trajectory_xy = trajectory[..., :2]
        vel_xy = self.trajectory_xy[:, 1] - self.trajectory_xy[:, 0]
        
        velocity_command, self.policy_velocity_controller = velocity_from_local_command(self.cfg, vel_xy)
        self.policy_joint_velocities = self.differential_controller.forward_torch(self.policy_velocity_controller)
        
        self.command_actions = self.policy_actions
    
    def apply_actions(self):
        self._asset.set_joint_velocity_target(self.joint_velocities, joint_ids=self._joint_ids)
    
    @property
    def action_dim(self) -> int:
        return self.cfg.action_steps * 3
