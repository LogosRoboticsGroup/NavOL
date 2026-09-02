import math
import random
import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners import materials
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns, CameraCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import navol.tasks.manager_based.navdp.mdp as mdp
from isaaclab.terrains import TerrainImporterCfg
from navol.assets import DINGO_CFG, DINGO_CameraCfg, DINGO_ContactCfg, DINGO_WHEEL_JOINTS, DINGO_BASE_LINK, DINGO_THRESHOLD, DINGO_MetricCameraCfg


GOAL_CFG = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Goal",\
    spawn = sim_utils.SphereCfg(visual_material=materials.PreviewSurfaceCfg(diffuse_color=(1.0,0.0,0.0)),visible=False,radius=0.25),
)

BENCH_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/Scene",
    terrain_type="usd",
    usd_path=f"",
)

@configclass
class ExplorationSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = BENCH_TERRAIN_CFG
    robot: ArticulationCfg = DINGO_CFG
    contact_sensor: ContactSensorCfg = DINGO_ContactCfg
    camera_sensor: CameraCfg = DINGO_CameraCfg
    metric_sensor: CameraCfg = DINGO_MetricCameraCfg

@configclass
class ExploreObservationsCfg:
    """Observation specifications for the MDP."""
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

    @configclass
    class PolicyCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func = mdp.RGBD_feature,
            # params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )

    # TODO: update occupancy pcd

    # @configclass
    # class MetricRGBImageCfg(ObsGroup):
    #     depth_measurement = ObsTerm(
    #         func = mdp.camera_rgb_data,
    #         params = {'asset_cfg':SceneEntityCfg("metric_sensor")}
    #     )
    # @configclass
    # class MetricDepthImageCfg(ObsGroup):
    #     depth_measurement = ObsTerm(
    #         func = mdp.camera_depth_data,
    #         params = {'asset_cfg':SceneEntityCfg("metric_sensor")}
    #     )
    
    policy: PolicyCfg = PolicyCfg()
    # rgb: RGBDCfg = RGBDCfg()

    # metric_rgb: MetricRGBImageCfg=MetricRGBImageCfg()
    # metric_depth: MetricDepthImageCfg=MetricDepthImageCfg()

@configclass
class DingoActionsCfg:
    joint_vel = mdp.NavDPMPCActionCfg(asset_name="robot", 
                                      joint_names=DINGO_WHEEL_JOINTS, 
                                      scale=1.0, 
                                      use_default_offset=True, 
                                      debug_vis=True, 
                                      action_steps=24)

@configclass
class DingoExploreTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=DINGO_BASE_LINK), "threshold": DINGO_THRESHOLD},
    )
    struck = DoneTerm(func=mdp.stuck_terminal_check,
                      params={"robot_asset_cfg": SceneEntityCfg("robot"), 
                              "window_size": 30, 
                              "threshold": 0.1})

@configclass
class ExploreEventCfg:
    """Configuration for events.""" 
    reset_pose = EventTerm(func=mdp.exploration_reset,
                           mode='reset',
                           params={"height_offset":0.1,
                                   "robot_visible": False,
                                   "light_enabled": False})

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    task_command = mdp.TaskCommandCfg(
        resampling_time_range=(1e5, 1e5),
        task_prob=[0.0, 0.0, 1.0, 0.0], # nogoal
    )

@configclass
class DingoExplorationCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = ExplorationSceneCfg(env_spacing=0.0)
    observations = ExploreObservationsCfg()
    actions = DingoActionsCfg()
    terminations = DingoExploreTerminationsCfg()
    events = ExploreEventCfg()
    rewards = None
    command = CommandsCfg()
    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True