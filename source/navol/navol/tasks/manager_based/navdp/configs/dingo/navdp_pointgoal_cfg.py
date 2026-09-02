import math
import random
import torch
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import MISSING
from typing import Literal
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners import materials
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns, CameraCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RecorderTermCfg as RecorderTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import navol.tasks.manager_based.navdp.mdp as mdp
from isaaclab.terrains import TerrainImporterCfg
from navol.assets import DINGO_CFG, DINGO_CameraCfg, DINGO_ContactCfg, DINGO_WHEEL_JOINTS
from navol.training import CANONICAL_TRAINING


GOAL_CFG = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Goal",\
    spawn = sim_utils.SphereCfg(visual_material=materials.PreviewSurfaceCfg(diffuse_color=(1.0,0.0,0.0)),visible=False,radius=0.25),
)

BENCH_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/Scene",
    terrain_type="usd",
    usd_path=f"",
)

APARTMENT_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/Scene2",
    terrain_type="usd",
    usd_path="",
)

# GROUND_CFG = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Ground",\
# GROUND_CFG = AssetBaseCfg(prim_path="/World/Ground",\
#     spawn = sim_utils.GroundPlaneCfg(),
# )

@configclass
class PointNavSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = BENCH_TERRAIN_CFG
    # terrain2: TerrainImporterCfg = APARTMENT_TERRAIN_CFG
    # terrain_collection: RigidObjectCollectionCfg = BENCH_TERRAIN_CFG_All
    robot: ArticulationCfg = DINGO_CFG
    contact_sensor: ContactSensorCfg = DINGO_ContactCfg
    camera_sensor: CameraCfg = DINGO_CameraCfg
    goal: AssetBaseCfg = GOAL_CFG

@configclass
class PointNavObservationsCfg:
    # """Observation specifications for the MDP."""
    # @configclass
    # class PolicyCfg(ObsGroup):
    #     """Observations for policy group."""
    #     # observation terms (order preserved)
    #     base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    #     base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    #     base_pos = ObsTerm(func=mdp.root_pos_w)
    #     base_rot = ObsTerm(func=mdp.root_quat_w)
    #     def __post_init__(self):
    #         self.enable_corruption = True
    #         self.concatenate_terms = True
    
    # @configclass
    # class RGBDCfg(ObsGroup):
    #     rgb_measurement = ObsTerm(
    #         func = mdp.RGBD_feature,
    #         # params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
    #     )

    # @configclass
    # class GoalPoseCfg(ObsGroup):
    #     pose_measurement = ObsTerm(
    #         func = mdp.oracle_imu_pose_data,
    #         params = {'robot_asset_cfg':SceneEntityCfg("robot")}
    #     )
        
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_rot = ObsTerm(func=mdp.root_quat_w)

        rgbd_token = mdp.RGBDFeatureCfg(
            func = mdp.RGBD_feature,
            image_size = 224,
            token_dim = 384,
            memory_size = 8
        )

        rgb = ObsTerm(
            func = mdp.rgb_only,
        )

        depth = ObsTerm(
            func = mdp.depth_only,
        )

        goal_pose = ObsTerm(
            func = mdp.oracle_imu_pose_data,
            params = {'robot_asset_cfg':SceneEntityCfg("robot")}
        )

        goal_w = ObsTerm(
            func = mdp.goal_pos_w,
        )
        
    policy: PolicyCfg = PolicyCfg(concatenate_terms=False)
    # rgbd: RGBDCfg = RGBDCfg()
    # goal_pose: GoalPoseCfg = GoalPoseCfg()

@configclass
class DingoActionsCfg:
    # joint_vel = mdp.NavDPMPCActionCfg(asset_name="robot", 
    #                                   joint_names=DINGO_WHEEL_JOINTS, 
    #                                   scale=1.0, 
    #                                   use_default_offset=True, 
    #                                   debug_vis=False, 
    #                                   apply_actions=True,
    #                                   action_steps=24)
    # joint_command = mdp.VelocityCommandActionCfg(asset_name="robot", 
    #                                   joint_names=DINGO_WHEEL_JOINTS, 
    #                                   mesh_path=None,
    #                                   scale=1.0, 
    #                                   use_default_offset=True, 
    #                                   apply_actions=False,
    #                                   debug_vis=False)
    joint_combined = mdp.VelocityCombinedActionCfg(
                                      asset_name="robot",
                                      joint_names=DINGO_WHEEL_JOINTS,
                                      scale=1.0,
                                      use_default_offset=True,
                                      debug_vis=False,
                                      action_steps=24,
                                      mesh_paths=None,
                                      use_mpc=CANONICAL_TRAINING.use_mpc,
                                      action_type="rand",
                                      action_rand_p=CANONICAL_TRAINING.policy_probability,
                                      critic_collision_reduction=CANONICAL_TRAINING.critic_collision_reduction,
                                      critic_safe_distance=CANONICAL_TRAINING.critic_safe_distance,
                                      critic_progress_alpha=CANONICAL_TRAINING.critic_progress_alpha,
                                      navmesh_radius=CANONICAL_TRAINING.navmesh_radius,)


@configclass
class DingoActionsEvalCfg:
    joint_combined = mdp.VelocityActionEvalCfg(
                                      asset_name="robot",
                                      joint_names=DINGO_WHEEL_JOINTS,
                                      scale=1.0,
                                      use_default_offset=True,
                                      use_mpc=False,
                                      debug_vis=False,
                                      action_steps=24,)
@configclass
class PointNavTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(func=mdp.fall_check,
                      params={"robot_asset_cfg": SceneEntityCfg("robot"), 
                              "window_size": 10, 
                              "threshold": 1})
    arrive_goal = DoneTerm(func=mdp.arrival_terminal_check,
                           params={"robot_asset_cfg":SceneEntityCfg("robot"), "distance_threshold":1.0, "velocity_threshold":0.5})
    stuck = DoneTerm(func=mdp.stuck_terminal_check,
                      params={"robot_asset_cfg": SceneEntityCfg("robot"), 
                              "window_size": 30, 
                              "threshold": 0.1})

@configclass
class PointNavEventCfg:
    """Configuration for events.""" 
    reset_pose = EventTerm(func=mdp.pointnav_reset,
                           mode='reset',
                           params={"height_offset":0.1,
                                   "robot_visible": False,
                                   "light_enabled": False,
                                   "sample_from_npy": False,
                                   "rand_camera": True,
                                   "height_range": CANONICAL_TRAINING.camera_height_range,
                                   "pitch_range": CANONICAL_TRAINING.camera_pitch_range,})

@configclass
class PointNavEvalEventCfg:
    """Configuration for events.""" 
    reset_pose = EventTerm(func=mdp.pointnav_eval_reset,
                           mode='reset',
                           params={"height_offset":0.1,
                                   "robot_visible": False,
                                   "light_enabled": False,})

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # reach_goal = RewTerm(
    #     func=mdp.reach_goal,
    #     weight=0.5,
    #     params={"threshold": 0.35},
    # )
    
    # goal_dis = RewTerm(
    #     func=mdp.goal_dis, weight=5.0
    # )
    # goal_dis_z = RewTerm(
    #     func=mdp.goal_dis_z, weight=30.0
    # )
    # goal_heading = RewTerm(
    #     func=mdp.goal_heading_l1, weight=0.3
    # )
    # stand_still_at_goal = RewTerm(
    #     func=mdp.stand_still_at_goal, weight=1.0
    # )
    # track_lin_vel_xy_exp_command = RewTerm(
    #     func=mdp.track_lin_vel_xy_exp_command, weight=0.2, params={"std": math.sqrt(0.25)}
    # )
    # track_ang_vel_z_exp_command = RewTerm(
    #     func=mdp.track_ang_vel_z_exp_command, weight=0.2, params={"std": math.sqrt(0.25)}
    # )
    # action_l2 = RewTerm(func=mdp.action_l2, weight=-0.002)

    # track_predict_velocity_command = RewTerm(
    #     func=mdp.track_predict_velocity_command, weight=0.01, params={"action": "joint_vel", "action_nav": "joint_command", "std": 1.0}
    # )
    track_predict_trajectory_command = RewTerm(
        func=mdp.track_predict_trajectory_command, weight=0.2, params={"std": 1.0}
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    task_command = mdp.TaskCommandCfg(
        resampling_time_range=(1e5, 1e5),
        task_prob=[1.0, 0.0, 0.0, 0.0], # pointgoal
    )

# @configclass
# class Recordscfg:
#     info = mdp.RewardRecorderManagerCfg()

@configclass
class DingoPointNavCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = PointNavSceneCfg(num_envs=2, env_spacing=0.0)
    observations = PointNavObservationsCfg()
    actions = DingoActionsCfg()
    terminations = PointNavTerminationsCfg()
    events = PointNavEventCfg()
    rewards = RewardsCfg()
    command = CommandsCfg()
    sim = sim_utils.SimulationCfg(
        render=sim_utils.RenderCfg(
            enable_translucency=True,
            enable_reflections=True,
            enable_global_illumination=True,
            antialiasing_mode="DLAA",
            enable_dlssg=True,
            dlss_mode=0,
            enable_dl_denoiser=True,
            enable_direct_lighting=True,
            samples_per_pixel=4,
            enable_shadows=True,
            enable_ambient_occlusion=True,
        )
    )
    # recorders = Recordscfg()
    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True


@configclass
class DingoPointNavCfg_Play(DingoPointNavCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class DingoPointNavCfg_Eval(DingoPointNavCfg):
    actions = DingoActionsEvalCfg()
    events = PointNavEvalEventCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 1
