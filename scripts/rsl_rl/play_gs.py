"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import numpy as np
import torch

import carb.input
import gymnasium as gym
import omni.appwindow

# Import extensions to set up environment tasks
import navol.tasks  # noqa: F401
from carb.input import KeyboardEventType
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from navol.wrapper import RslRlGSEnvWrapper, RslRlOnPolicyRunnerCfg

from rsl_rl.runners import OnPolicyRunner

MOVE_CAMERA = False


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlGSEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # if "unitree_go2_gs" not in log_root_path:
    if args_cli.task == "go2_gs_play" or args_cli.task == "go2_gs_nav_play":
        ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )
    # camera_direction = [2, 2, -4]
    camera_direction = [5, 0, -6]

    def on_keyboard_input(e):
        if e.input == carb.input.KeyboardInput.W:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                env.override_command[:, 0] = 0.8
        if e.input == carb.input.KeyboardInput.S:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                env.override_command[:, 0] = -0.8
        if e.input == carb.input.KeyboardInput.A:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                env.override_command[:, 1] = 0.5
        if e.input == carb.input.KeyboardInput.D:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                env.override_command[:, 1] = -0.5
        if e.input == carb.input.KeyboardInput.F:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                env.override_command[:, 2] = 1
        if e.input == carb.input.KeyboardInput.G:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                env.override_command[:, 2] = -1
        if e.input == carb.input.KeyboardInput.X:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                env.override_command[:] = 0
        if e.input == carb.input.KeyboardInput.N:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                env.env.scene.terrain.terrain_types[:] -= 1
                env.env.scene.terrain.terrain_types[:] = torch.clip(env.env.scene.terrain.terrain_types[:], 0, 19)
        if e.input == carb.input.KeyboardInput.M:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                env.env.scene.terrain.terrain_types[:] += 1
                env.env.scene.terrain.terrain_types[:] = torch.clip(env.env.scene.terrain.terrain_types[:], 0, 19)

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input = carb.input.acquire_input_interface()
    keyboard_sub = input.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    camera_follow_id = 0

    camera_position = env.unwrapped.scene.env_origins[camera_follow_id].cpu().numpy() - camera_direction + [1.8, -0.8, 0]
    env.sim.set_camera_view(camera_position, camera_position + camera_direction)
    
    # from pxr import UsdGeom
    # import trimesh
    # stage = env.unwrapped.scene.stage
    # prims = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    # vertices = np.array(UsdGeom.Mesh(prims[0]).GetPointsAttr().Get())
    # vertices = vertices + (3.2, 0.0, -0.01)
    # faces = np.array(UsdGeom.Mesh(prims[0]).GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    # mesh_object=trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # mesh_object.export("object.ply")
    # vertices = np.array(UsdGeom.Mesh(prims[-1]).GetPointsAttr().Get())
    # faces = np.array(UsdGeom.Mesh(prims[-1]).GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    # verts_f = vertices[faces]
    # mask = ((-0.2 < verts_f[..., 0]) & (verts_f[..., 0] < 3.2) & (-3.5 < verts_f[..., 1]) & (verts_f[..., 1] < 1.9)).all(axis=-1)
    # mesh_ground=trimesh.Trimesh(vertices=vertices, faces=faces[mask], process=False)
    # mesh_ground.remove_unreferenced_vertices()
    # mesh_ground = mesh_ground.copy()
    # mesh_ground.export("ground.ply")
    # mesh = trimesh.util.concatenate([mesh_object, mesh_ground])
    # mesh.vertices = mesh.vertices[:, [1, 2, 0]]
    # mesh.visual.vertex_colors = np.array([[255, 0, 0, 255]] * len(mesh.vertices), dtype=np.uint8)
    # mesh.export("ckpt/combined.ply")
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if MOVE_CAMERA:
            camera_position = env.root_states[camera_follow_id, :3].cpu().numpy() - camera_direction
            env.sim.set_camera_view(camera_position, camera_position + camera_direction)
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
