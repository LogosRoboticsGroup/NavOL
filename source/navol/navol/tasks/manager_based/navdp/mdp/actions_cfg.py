from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.markers import VisualizationMarkersCfg
from navol.assets import DINGO_WHEEL_RADIUS, DINGO_WHEEL_BASE
from navol.training import CANONICAL_TRAINING

from . import actions


@configclass
class NavDPMPCActionCfg(JointVelocityActionCfg):
    """Configuration for the NavDP MPC action term.

    See :class:`NavDPMPCAction` for more details.
    """

    class_type: type[ActionTerm] = actions.NavDPMPCAction

    use_default_offset: bool = True

    wheel_radius: float = DINGO_WHEEL_RADIUS

    wheel_base: float = DINGO_WHEEL_BASE

    speed: float = 0.5

    max_linear_speed: float = 0.5

    vel_scale: float = 3.0
    
    apply_actions: bool = True

    max_angular_speed: float = 0.5

    action_steps: int = MISSING

@configclass
class VelocityCommandActionCfg(JointVelocityActionCfg):
    """Configuration for the velocity command action term.

    See :class:`NavDPMPCAction` for more details.
    """

    class_type: type[ActionTerm] = actions.VelocityCommandAction

    use_default_offset: bool = True

    wheel_radius: float = DINGO_WHEEL_RADIUS

    wheel_base: float = DINGO_WHEEL_BASE

    speed: float = 0.5

    max_linear_speed: float = 0.5

    max_angular_speed: float = 0.5
    
    vel_scale: float = 3.0

    apply_actions: bool = True
    
    predict_size: int = 24

    mesh_path: str = MISSING

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

@configclass
class VelocityCombinedActionCfg(JointVelocityActionCfg):

    class_type: type[ActionTerm] = actions.VelocityCombinedAction

    use_default_offset: bool = True

    wheel_radius: float = DINGO_WHEEL_RADIUS

    wheel_base: float = DINGO_WHEEL_BASE

    speed: float = 0.5

    max_linear_speed: float = 0.5

    vel_scale: float = 2.0
    
    angular_scale: float = 1.0
    
    factor_scale_vel: float = 2.0
    
    factor_scale_ang: float = 4.0
    
    max_angular_speed: float = 0.5

    predict_size: int = 24

    action_steps: int = 24

    mesh_paths: list = MISSING
    
    action_type: str = "rand"
    
    action_rand_p: float = CANONICAL_TRAINING.policy_probability

    # ``mean`` preserves existing checkpoint semantics. ``sum`` matches the
    # collision-count term as written in the paper and may require retraining.
    critic_collision_reduction: str = CANONICAL_TRAINING.critic_collision_reduction

    critic_safe_distance: float = CANONICAL_TRAINING.critic_safe_distance

    critic_progress_alpha: float = CANONICAL_TRAINING.critic_progress_alpha
    
    scene_scale: float = 1.0
    
    navmesh_radius: float = CANONICAL_TRAINING.navmesh_radius
    
    search_radius: float = CANONICAL_TRAINING.local_search_radius
    
    use_mpc: bool = CANONICAL_TRAINING.use_mpc
    

@configclass
class VelocityActionEvalCfg(JointVelocityActionCfg):

    class_type: type[ActionTerm] = actions.VelocityActionEval

    use_default_offset: bool = True

    wheel_radius: float = DINGO_WHEEL_RADIUS

    wheel_base: float = DINGO_WHEEL_BASE

    speed: float = 0.5

    max_linear_speed: float = 0.5

    vel_scale: float = 2.0
    
    angular_scale: float = 1.0
    
    factor_scale_vel: float = 2.0
    
    factor_scale_ang: float = 4.0
    
    max_angular_speed: float = 0.5

    predict_size: int = 24

    action_steps: int = 24
    
    use_mpc: bool = False
