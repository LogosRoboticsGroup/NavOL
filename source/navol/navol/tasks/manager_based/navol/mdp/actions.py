from __future__ import annotations

import os
import math
import cv2
import trimesh
# import habitat_sim.utils.common
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
import omni.log
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

from . import actions_cfg
# import habitat_sim
# from habitat.utils.visualizations import maps
# import magnum as mn
from isaaclab.utils.math import quat_apply, quat_inv
from .utils.path_smoothing import get_smooth_points


class JointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self._num_joints, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


class VelocityCommandAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    """The configuration of the action term."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset

        policy_cfg = {
            "class_name": "ActorCriticRecurrent",
            "init_noise_std": 0.75,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        }

        actor_critic_class = eval(policy_cfg["class_name"])  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(48, 235, 12, **policy_cfg).to(
            self._env.device
        )
        actor_critic.load_state_dict(torch.load(cfg.policy_dir)["model_state_dict"])
        actor_critic.eval()
        self.policy = actor_critic.act_inference
        self.velocity_range = torch.tensor(cfg.velocity_range).cuda()
        self.velocity_command = torch.zeros([self.num_envs, 3]).cuda()
        self.uplevel_frequency = cfg.uplevel_frequency
        self.lowlevel_counter = 0
        self.tanh = torch.nn.Tanh()
        
        self.mesh_path = cfg.mesh_path
        
        sim_settings = {
            # "scene": cfg.mesh_path,  # Scene path
            "scene": "ckpt/combined2.ply",  # Scene path
            "default_agent": 0,  # Index of the default agent
            "sensor_height": 0.5,  # Height of sensors in meters, relative to the agent
            "width": 256,  # Spatial resolution of the observations
            "height": 256,
        }

        if self.mesh_path is not None:
            self.sim = habitat_sim.Simulator(self.make_simple_cfg(sim_settings))
            agent = self.sim.initialize_agent(sim_settings["default_agent"])
            agent_state = habitat_sim.AgentState()
            # agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
            # agent_state.position = np.array([0.899869, -1.157283, 0.05])  # in world space
            # agent_state.position = np.array([2.922204, -0.432507, 1.5])  # in world space
            agent_state.position = np.array([-2.402628, 0.01, 1.495310])  # in world space
            agent_state.rotation = np.array([0.0, 1.0, 0.0, 0.0])  # in world space
            # agent_state.position = np.array([1., 0.0, 1.])  # in world space
            agent.set_state(agent_state)

            # Get agent state
            agent_state = agent.get_state()
            print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
            
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.cell_size = 0.01
            navmesh_settings.cell_height = 0.02
            navmesh_settings.agent_height = 0.3
            navmesh_settings.agent_radius = 0.35
            navmesh_settings.agent_max_climb = 0.05
            navmesh_settings.agent_max_slope = 45
            navmesh_settings.edge_max_len = navmesh_settings.agent_radius * 2.0

            navmesh_success = self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
            print("The NavMesh bounds are: " + str(self.sim.pathfinder.get_bounds()))
        
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_command_b[:, 1] = 1.0
    
    def make_simple_cfg(self, settings):
        # simulator backend
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = settings["scene"]

        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
        # agent

        # In the 1st example, we attach only one sensor,
        # a RGB visual sensor, to the agent
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
        rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

        agent_cfg.sensor_specifications = [rgb_sensor_spec]

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def process_actions(self, actions: torch.Tensor):
        self.velocity_command = self.tanh(actions) * self.velocity_range
        return
        
        
        rgb_commands = self._env.command_manager.get_command("rgb_command")
        robot_pos = self._env.scene["robot"].data.root_pos_w - self._env.scene.env_origins
        robot_quat = self._env.scene["robot"].data.root_quat_w
        robot_quat_inv = quat_inv(robot_quat)

        goal_red = self._env.scene["cone_red"].data.object_pos_w.squeeze(1) - self._env.scene.env_origins
        goal_green = self._env.scene["cone_green"].data.object_pos_w.squeeze(1) - self._env.scene.env_origins
        goal_blue = self._env.scene["cone_blue"].data.object_pos_w.squeeze(1) - self._env.scene.env_origins
        
        robot_pos = robot_pos[:, [1, 2, 0]]
        goal_red = goal_red[:, [1, 2, 0]]
        goal_green = goal_green[:, [1, 2, 0]]
        goal_blue = goal_blue[:, [1, 2, 0]]

        goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
        pos_goal = (goal_tensor * rgb_commands.unsqueeze(-1)).sum(1)
        
        robot_pos = robot_pos.cpu().numpy()
        pos_goal = pos_goal.cpu().numpy()
        
        path_points = []
        for i in range(self.num_envs):
            path = habitat_sim.ShortestPath()
            path.requested_start = robot_pos[i]
            path.requested_end = pos_goal[i]

            found_path = self.sim.pathfinder.find_path(path)
            geodesic_distance = path.geodesic_distance
            path.points.extend(path.requested_end)
            
            if len(path.points) < 2:
                sample_points = np.array([[0.,0.,0.],[0.,0.,0.]])
            else:
                sample_points = np.array(path.points)
            
            # sample_points = smooth_path_bezier(path_points, num_samples=20)
            sample_points = get_smooth_points(sample_points, path_finder=self.sim.pathfinder, num_samples=20)
            if len(sample_points) < 2:
                sample_points = np.concatenate((sample_points, [pos_goal[i]]))

            path_points.append(np.array(sample_points)[:2, [2, 0, 1]])
        path_points = np.stack(path_points, axis=0)
        # self.path_points = torch.from_numpy(path_points).to(self.device).float()
        self.path_points = torch.tensor(path_points, device=self.device, dtype=torch.float32)
        # print("geodesic_distance : " + str(geodesic_distance))
        # print("path_points : " + str(path_points))
        vel_navmesh = (self.path_points[:, 1] - self.path_points[:, 0])
        
        base_quat_w = self.robot.data.root_quat_w
        base_quat_inv_w = math_utils.quat_inv(base_quat_w)
        velocity_command = math_utils.quat_apply(base_quat_inv_w, vel_navmesh)
        velocity_command_xy = velocity_command[:, :2]
        velocity_command_xy = torch.nn.functional.normalize(velocity_command_xy, dim=-1)
        velocity_command_xy[..., 0].clamp_max_(self.velocity_range[0])
        velocity_command_xy[..., 1].clamp_max_(self.velocity_range[1])
        # self.velocity_command = velocity_command

        heading_angle = torch.atan2(velocity_command_xy[..., 1], velocity_command_xy[..., 0])[..., None]
        velocity_command_w = (heading_angle).clamp_max(self.velocity_range[2])
        
        angle_threshold = torch.pi / 6
        
        angle_mask = torch.abs(heading_angle) > angle_threshold
        angle_mask = angle_mask.squeeze(-1)
        angle_factor = torch.cos(torch.clamp(torch.abs(heading_angle), 0, torch.pi/2))
        angle_factor = torch.clamp(angle_factor, 0.2, 1.0)

        adjusted_velocity_xy = velocity_command_xy.clone()
        adjusted_velocity_xy[angle_mask] = velocity_command_xy[angle_mask] * angle_factor[angle_mask]
        self.velocity_command_nav = torch.cat([adjusted_velocity_xy, velocity_command_w], dim=-1)

        # self.velocity_command_nav = torch.where(
        #     heading_angle > 3.14/180 * 10,
        #     torch.cat([torch.zeros_like(velocity_command_xy), velocity_command_w], dim=-1),
        #     torch.cat([velocity_command_xy, velocity_command_w], dim=-1),
        # )

        # self.velocity_command = torch.cat([velocity_command_xy, velocity_command_w], dim=-1)
        # TODO: 有的点离得太近导致速度过小

    def apply_actions(self):
        if self.lowlevel_counter % 4 == 0:
            obs = self.get_observations()
            with torch.inference_mode():
                obs[:, 9:12] = self.velocity_command
                # obs[:, 9:12] = self.velocity_command_nav
                joint_positions = self.policy(obs)
            super().process_actions(joint_positions)
        self.lowlevel_counter += 1
        self.lowlevel_counter %= 4
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def get_observations(self):
        """Returns the current observations of the environment."""
        obs = self._env.observation_manager.compute_group("locomotion")
        return obs

    @property
    def action_dim(self) -> int:
        return 3
    
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
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # # -- resolve the scales and quaternions
        vel_navmesh = (self.path_points[:, 1] - self.path_points[:, 0])
        heading_angle = torch.atan2(vel_navmesh[..., 1], vel_navmesh[...,0])
        zeros = torch.zeros_like(heading_angle)
        vel_des_arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        
        vel_des_arrow_scale, _ = self._resolve_xy_velocity_to_arrow(torch.nn.functional.normalize(self.vel_command_b[:, :2], dim=-1))
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(torch.nn.functional.normalize(self.robot.data.root_lin_vel_b[:, :2], dim=-1))
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)


class VelocityCommandDPAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.VelocityCommandActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset

        policy_cfg = {
            "class_name": "ActorCriticRecurrent",
            "init_noise_std": 0.75,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        }

        actor_critic_class = eval(policy_cfg["class_name"])  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(48, 235, 12, **policy_cfg).to(
            self._env.device
        )
        policy_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.policy_dir))
        actor_critic.load_state_dict(torch.load(policy_dir)["model_state_dict"])
        actor_critic.eval()
        self.policy = actor_critic.act_inference
        self.velocity_range = torch.tensor(cfg.velocity_range).cuda()
        self.velocity_command = torch.zeros([self.num_envs, 3]).cuda()
        self.uplevel_frequency = cfg.uplevel_frequency
        self.lowlevel_counter = 0
        self.highlevel_counter = 0
        self.tanh = torch.nn.Tanh()

    def process_actions(self, actions: torch.Tensor):
        # self.velocity_command = self.tanh(actions.reshape(actions.shape[0], self.cfg.action_steps, -1)) * self.velocity_range
        self.velocity_command = actions.reshape(actions.shape[0], self.cfg.action_steps, -1)
        self.highlevel_counter = 0

    def apply_actions(self):
        if self.lowlevel_counter % 4 == 0:
            obs = self.get_observations()
            with torch.inference_mode():
                obs[:, 9:12] = self.velocity_command[:, self.highlevel_counter]
                self.highlevel_counter += 1
                joint_positions = self.policy(obs)
            super().process_actions(joint_positions)
        self.lowlevel_counter += 1
        self.lowlevel_counter %= 4
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def get_observations(self):
        """Returns the current observations of the environment."""
        obs = self._env.observation_manager.compute_group("locomotion")
        return obs

    @property
    def action_dim(self) -> int:
        return 3 * self.cfg.action_steps


class DummyAction(ActionTerm):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    """The configuration of the action term."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_command_b[:, 1] = 1.0
        
        self.mesh_path = cfg.mesh_path
        
        sim_settings = {
            # "scene": cfg.mesh_path,  # Scene path
            "scene": "ckpt/combined2.ply",  # Scene path
            "default_agent": 0,  # Index of the default agent
            "sensor_height": 0.5,  # Height of sensors in meters, relative to the agent
            "width": 256,  # Spatial resolution of the observations
            "height": 256,
        }
        self.sim = habitat_sim.Simulator(self.make_simple_cfg(sim_settings))
        agent = self.sim.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        # agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
        # agent_state.position = np.array([0.899869, -1.157283, 0.05])  # in world space
        # agent_state.position = np.array([2.922204, -0.432507, 1.5])  # in world space
        agent_state.position = np.array([-2.402628, 0.01, 1.495310])  # in world space
        agent_state.rotation = np.array([0.0, 1.0, 0.0, 0.0])  # in world space
        # agent_state.position = np.array([1., 0.0, 1.])  # in world space
        agent.set_state(agent_state)

        # Get agent state
        agent_state = agent.get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
        
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.cell_size = 0.01
        navmesh_settings.cell_height = 0.02
        navmesh_settings.agent_height = 0.3
        navmesh_settings.agent_radius = 0.3
        navmesh_settings.agent_max_climb = 0.2
        navmesh_settings.agent_max_slope = 45

        navmesh_success = self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
        print("The NavMesh bounds are: " + str(self.sim.pathfinder.get_bounds()))
        # top_down_map = maps.get_topdown_map(self.sim.pathfinder, height=self.sim.pathfinder.get_bounds()[0][1], meters_per_pixel=0.01)
        # recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        # top_down_map = recolor_map[top_down_map]
        # grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        # # convert world agent position to maps module grid point
        # agent_grid_pos = maps.to_grid(agent_state.position[2], agent_state.position[0], grid_dimensions, pathfinder=self.sim.pathfinder)
        # agent_forward = habitat_sim.utils.common.quat_to_magnum(self.sim.agents[0].get_state().rotation).transform_vector(mn.Vector3(0, 0, -1.0))
        # agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
        # maps.draw_agent(top_down_map, agent_grid_pos, agent_orientation, agent_radius_px=8)

        # rgb_commands = env.command_manager.get_command("rgb_command")
        # robot_pos = env.scene["robot"].data.root_pos_w
        # robot_quat = env.scene["robot"].data.root_quat_w
        # robot_quat_inv = quat_inv(robot_quat)

        # goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1)
        # goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1)
        # goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1)
        # goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
        # pos_goal = (goal_tensor * rgb_commands.unsqueeze(-1)).sum(1)
    
        # path = habitat_sim.ShortestPath()
        # path.requested_start = robot_pos.cpu().numpy()[0]
        # path.requested_end = pos_goal.cpu().numpy()[0]
    
    
        # sample1 = np.array([-0.598261, 0.342396, 2.496715])
        # sample2 = np.array([0.985114, -0.000741, 2.221107])
        # path = habitat_sim.ShortestPath()
        # path.requested_start = sample1
        # path.requested_end = sample2
        # found_path = self.sim.pathfinder.find_path(path)
        # geodesic_distance = path.geodesic_distance
        # path_points = path.points
        # for i in range(len(path_points)):
        #     agent_grid_pos = maps.to_grid(
        #         path_points[i][2], path_points[i][0], grid_dimensions, pathfinder=self.sim.pathfinder
        #     )
        #     maps.draw_agent(
        #         top_down_map, agent_grid_pos, agent_orientation, agent_radius_px=8
        #     )
        # cv2.imwrite("top_down_map.png", top_down_map[..., [2, 1, 0]])
        # # cv2.imwrite("test2.png", self.sim.get_sensor_observations()["color_sensor"][..., [2, 1, 0, 3]])
        # # mesh = trimesh.load(self.mesh_path, process=False)
        # trimesh.Trimesh(np.array(path_points)).export("path.ply")

    def make_simple_cfg(self, settings):
        # simulator backend
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = settings["scene"]

        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # In the 1st example, we attach only one sensor,
        # a RGB visual sensor, to the agent
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
        rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

        agent_cfg.sensor_specifications = [rgb_sensor_spec]

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def process_actions(self, actions: torch.Tensor):
        # return
        # top_down_map = maps.get_topdown_map(self.sim.pathfinder, height=self.sim.pathfinder.get_bounds()[0][1], meters_per_pixel=0.01)
        # recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        # top_down_map = recolor_map[top_down_map]
        # grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])

        rgb_commands = self._env.command_manager.get_command("rgb_command")
        robot_pos = self._env.scene["robot"].data.root_pos_w - self._env.scene.env_origins
        robot_quat = self._env.scene["robot"].data.root_quat_w
        robot_quat_inv = quat_inv(robot_quat)

        goal_red = self._env.scene["cone_red"].data.object_pos_w.squeeze(1) - self._env.scene.env_origins
        goal_green = self._env.scene["cone_green"].data.object_pos_w.squeeze(1) - self._env.scene.env_origins
        goal_blue = self._env.scene["cone_blue"].data.object_pos_w.squeeze(1) - self._env.scene.env_origins
        
        robot_pos = robot_pos[:, [1, 2, 0]]
        goal_red = goal_red[:, [1, 2, 0]]
        goal_green = goal_green[:, [1, 2, 0]]
        goal_blue = goal_blue[:, [1, 2, 0]]

        goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
        pos_goal = (goal_tensor * rgb_commands.unsqueeze(-1)).sum(1)
        
        # maps.draw_agent(top_down_map, maps.to_grid(goal_red.cpu().numpy()[0][2], goal_red.cpu().numpy()[0][0], grid_dimensions, pathfinder=self.sim.pathfinder), 0.0, agent_radius_px=8)
        # maps.draw_agent(top_down_map, maps.to_grid(goal_green.cpu().numpy()[0][2], goal_green.cpu().numpy()[0][0], grid_dimensions, pathfinder=self.sim.pathfinder), 0.0, agent_radius_px=8)
        # maps.draw_agent(top_down_map, maps.to_grid(goal_blue.cpu().numpy()[0][2], goal_blue.cpu().numpy()[0][0], grid_dimensions, pathfinder=self.sim.pathfinder), 0.0, agent_radius_px=8)
        # maps.draw_agent(top_down_map, maps.to_grid(robot_pos.cpu().numpy()[0][2], robot_pos.cpu().numpy()[0][0], grid_dimensions, pathfinder=self.sim.pathfinder), 0.0, agent_radius_px=8)
        
        path_points = []
        for i in range(self.num_envs):
            path = habitat_sim.ShortestPath()
            path.requested_start = robot_pos.cpu().numpy()[i]
            path.requested_end = pos_goal.cpu().numpy()[i]

            found_path = self.sim.pathfinder.find_path(path)
            geodesic_distance = path.geodesic_distance
            path_points.append(np.array(path.points)[:, [2, 0, 1]])
        path_points = np.stack(path_points, axis=0)
        self.path_points = torch.from_numpy(path_points).to(self.device).float()
        print("geodesic_distance : " + str(geodesic_distance))
        print("path_points : " + str(path_points))
        # for i in range(len(path_points)):
        #     agent_grid_pos = maps.to_grid(
        #         path_points[i][2], path_points[i][0], grid_dimensions, pathfinder=self.sim.pathfinder
        #     )
        #     maps.draw_agent(
        #         top_down_map, agent_grid_pos, 0.0, agent_radius_px=8
        #     )
        # cv2.imwrite("top_down_map.png", top_down_map[..., [2, 1, 0]])
        # trimesh.Trimesh(np.array(path_points)).export("path.ply")

    def apply_actions(self):
        pass

    @property
    def action_dim(self) -> int:
        return 0
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return 0

    @property
    def processed_actions(self) -> torch.Tensor:
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
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # # -- resolve the scales and quaternions
        vel_navmesh = (self.path_points[:, 1] - self.path_points[:, 0])
        heading_angle = torch.atan2(vel_navmesh[..., 1], vel_navmesh[...,0])
        zeros = torch.zeros_like(heading_angle)
        vel_des_arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        
        vel_des_arrow_scale, _ = self._resolve_xy_velocity_to_arrow(torch.nn.functional.normalize(self.vel_command_b[:, :2], dim=-1))
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(torch.nn.functional.normalize(self.robot.data.root_lin_vel_b[:, :2], dim=-1))
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
