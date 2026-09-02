"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--scene_dir", type=str, default=None,
    help="Scene dataset root (default: <asset-root>/datasets/train/3d_front_scene_50).")
parser.add_argument(
    "--checkpoint_path", type=str, default=None,
    help=("Raw NavOL policy state dict (default: "
          "<asset-root>/models/checkpoints/navol-mpc-iter1000.pt)."))
parser.add_argument(
    "--scene_scale", type=float, default=1.0)
parser.add_argument(
    "--num_episodes", type=int, default=100)
parser.add_argument(
    "--save", action="store_true", default=False)
parser.add_argument(
    "--use_navmesh", action="store_true", default=False)
parser.add_argument(
    "--use_mpc", action="store_true", default=False)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
args_cli.enable_cameras = True
# args_cli.headless = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import numpy as np
import torch
import csv
import json

import carb.input
import gymnasium as gym
import omni.appwindow

# Import extensions to set up environment tasks
import navol.tasks  # noqa: F401
from carb.input import KeyboardEventType
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
import isaaclab.sim as sim_utils
from navol.wrapper import RslRlDPEnvWrapper, RslRlOnPolicyRunnerCfg
from navol.terrains.utils import find_usd_path, adjust_usd_scale
from navol.evaluation.metrics import is_success
from navol.paths import dataset_path, policy_checkpoint_path
from isaaclab.managers import SceneEntityCfg
from rsl_rl.runners import NavdpRunner
from torchvision.utils import save_image
import imageio
from tqdm import tqdm
import random
from matplotlib import colormaps as cm
import cv2

seed = 42  # Choose your desired seed value
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

def write_metrics(metrics, path="exploration.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def main():
    """Play with RSL-RL agent."""
    scene_dir = str(dataset_path("train/3d_front_scene_50")) if args_cli.scene_dir is None else args_cli.scene_dir
    usd_path = os.path.join(scene_dir, "usd", "scene.usd")
    init_path = os.path.join(scene_dir, "sample_100.npy")
    index_file = os.path.join(scene_dir, "index.json")
    with open(index_file, "r") as f:
        scene_dict = json.load(f)
    
    if os.path.exists(os.path.join(scene_dir, "selected.json")):
        with open(os.path.join(scene_dir, "selected.json"), 'r') as f:
            selected_indices = json.load(f)
        selected_scene_names = [f"scene_{i:03d}" for i in selected_indices]
        scene_dict = {name: scene_dict[name] for name in selected_scene_names}
        print(f"Selected {len(selected_scene_names)} scenes.")
    
    mesh_paths = [os.path.join(scene_dir, "navmesh_scenes", scene_name + ".glb") for scene_name in scene_dict]
    scene = os.path.basename(os.path.normpath(scene_dir))
    print(f"[INFO] Loading Scene from USD: {usd_path}")
    print(f"[INFO] Loading Trajectory from NPY: {init_path}")
    print(f"[INFO] Loading Mesh from directory: {os.path.join(scene_dir, 'navmesh_scenes')}")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.scene.terrain.usd_path = usd_path
    # env_cfg.scene.terrain2.usd_path = apartment_path
    env_cfg.events.reset_pose.params["init_point_path"] = init_path
    env_cfg.actions.joint_combined.mesh_paths = mesh_paths
    env_cfg.actions.joint_combined.action_type = 'command' if args_cli.use_navmesh else 'policy'
    env_cfg.actions.joint_combined.scene_scale = args_cli.scene_scale
    env_cfg.actions.joint_combined.use_mpc = args_cli.use_mpc

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    checkpoint_path = policy_checkpoint_path(args_cli.checkpoint_path)
    agent_cfg.policy.pretrained_model_path = str(checkpoint_path)

    agent_cfg.seed = seed
    env_cfg.seed = agent_cfg.seed

    log_dir_scene = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, "play", scene))
    os.makedirs(log_dir_scene, exist_ok=True)
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlDPEnvWrapper(env)
    # adjust_usd_scale(scale=1)
    adjust_usd_scale(scale=args_cli.scene_scale)
    # adjust_usd_scale(scale=args_cli.scene_scale, rotation=(0.707, 0.707, 0, 0))

    print(f"[INFO]: Loading NavOL policy state dict from: {checkpoint_path}")
    ppo_runner = NavdpRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, infos = env.get_observations()
    episode_num = args_cli.num_envs - 1
    evaluation_metrics = []
    if args_cli.task != "nogoal":
        trajectory_length = np.zeros((env.num_envs))
        euclidean = np.sqrt(np.square(infos['observations']["policy"]['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))
    else:
        raise NotImplementedError
    
    progress_bars = [
        tqdm(total=int(env.max_episode_length), desc=f"Episode {i} (Env {i})", position=i, leave=False)
        for i in range(args_cli.num_envs)
    ]
    if args_cli.save:
        os.makedirs(os.path.join(log_dir_scene, "videos", ), exist_ok=True)
        fps_writer = [imageio.get_writer(os.path.join(log_dir_scene, "videos", f"traj_{i:03d}.mp4"), fps=10) for i in range(args_cli.num_envs)]
        print(f"[INFO] Saving videos to: {os.path.join(log_dir_scene, 'videos')}")

    # simulate environment
    while simulation_app.is_running():
        last_infos = infos.copy()
        with torch.inference_mode():
            actions = policy(obs)
            # actions = torch.zeros(args_cli.num_envs, 72, device=env.unwrapped.device)
            obs, _, dones, infos = env.step(actions)
            # root_pos = env.unwrapped.scene[SceneEntityCfg("robot").name].data.root_pos_w[0].tolist()
            # goal_pos = infos['observations']['policy']['goal_pose'][0].tolist()
            # print("root_pos_w: [{:.3f}, {:.3f}, {:.3f}] goal_pose: [{:.3f}, {:.3f}, {:.3f}]".format(
            #     root_pos[0], root_pos[1], root_pos[2], goal_pos[0], goal_pos[1], goal_pos[2]))
            # print("Camera pos:", env.unwrapped.scene[SceneEntityCfg("camera_sensor").name].data.pos_w)
        
            trajectory_length += infos['observations']["policy"]['base_lin_vel'].cpu().numpy()[:,0] * env.unwrapped.step_dt
            # print("Velocity:", infos['observations']["policy"]['base_lin_vel'].cpu().numpy())
            
            
        
        if args_cli.save:
            for i in range(args_cli.num_envs):
                # i=0
                trajectory_mask = env.unwrapped.scene[SceneEntityCfg("camera_sensor").name].data.output['rgb'].cpu().numpy()[i][..., [2,1,0]].copy()
                expert_trajectories = env.unwrapped.action_manager.get_term("joint_combined").path_points_local.cpu().numpy()[i][None]
                policy_trajectories = ppo_runner.alg.policy.all_trajectory.cpu().numpy()[i]
                critic_values = ppo_runner.alg.policy.critic_values.cpu().numpy()[i]
                sorted_indices = np.argsort(-critic_values, axis=0)
                k = 2
                topk_indice = sorted_indices[:k]
                policy_trajectories = policy_trajectories[topk_indice]
                critic_values = critic_values[topk_indice]
        
                camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0].cpu().numpy()
                
                def value_to_color(value, values_min, values_max):
                    fixed_min = -1.2
                    fixed_max = 0.2
                    value = np.clip(value, fixed_min, fixed_max)
                    normalized = (value - fixed_min) / (fixed_max - fixed_min)
                    if normalized < 0.5:
                        b = 255 * (1 - 2 * normalized)
                        g = 255 * (2 * normalized)
                        r = 0
                    else:
                        b = 0
                        g = 255 * (2 - 2 * normalized)
                        r = 255 * (2 * normalized - 1)
                    return (int(b), int(g), int(r))  # Return BGR color
                values_min = np.min(critic_values)
                values_max = np.max(critic_values)
                trajectory_colors = [value_to_color(v, values_min, values_max) for v in critic_values]
                for expert_waypoints in expert_trajectories:
                    # norm_value = np.clip(-value*0.1,0,1)
                    # norm_value = 1
                    # colormap = cm.get('jet')
                    # color = np.array(colormap(norm_value)[0:3]) * 255.0
                    color = np.array((211, 70, 140))
                    expert_input_points = np.zeros((expert_waypoints.shape[0],3)) - 0.2
                    expert_input_points[:,0:2] = expert_waypoints
                    expert_input_points[:,1] = -expert_input_points[:,1]
                    camera_z = trajectory_mask.shape[0] - 1 - camera_intrinsic[1][1] * expert_input_points[:,2] / (expert_input_points[:,0] + 1e-8) - camera_intrinsic[1][2]
                    camera_x = camera_intrinsic[0][0] * expert_input_points[:,1] / (expert_input_points[:,0] + 1e-8) + camera_intrinsic[0][2]
                    for j in range(camera_x.shape[0]-1):
                        try:
                            if camera_x[j] > 0 and camera_z[j] > 0 and camera_x[j+1] > 0 and camera_z[j+1] > 0:
                                trajectory_mask = cv2.line(trajectory_mask,(int(camera_x[j]),int(camera_z[j])),(int(camera_x[j+1]),int(camera_z[j+1])),(color.astype(np.uint8).tolist()),5)
                        except:
                            pass
                        # if camera_x[j] > 0 and camera_z[j] > 0 and camera_x[j+1] > 0 and camera_z[j+1] > 0 and camera_x[j] < trajectory_mask.shape[1] and camera_x[j+1] < trajectory_mask.shape[1] and camera_z[j] < trajectory_mask.shape[0] and camera_z[j+1] < trajectory_mask.shape[0]:
                        #     trajectory_mask = cv2.line(trajectory_mask,(int(camera_x[j]),int(camera_z[j])),(int(camera_x[j+1]),int(camera_z[j+1])),(color.astype(np.uint8).tolist()),5)
                for policy_waypoints,value, color in zip(policy_trajectories,critic_values, trajectory_colors):
                    # norm_value = np.clip(-value*0.1,0,1)
                    # colormap = cm.get('jet')
                    # color = np.array(colormap(norm_value)[0:3]) * 255.0
                    policy_input_points = np.zeros((policy_waypoints.shape[0],3)) - 0.2
                    policy_input_points[:,0:2] = policy_waypoints
                    policy_input_points[:,1] = -policy_input_points[:,1]
                    camera_z = trajectory_mask.shape[0] - 1 - camera_intrinsic[1][1] * policy_input_points[:,2] / (policy_input_points[:,0] + 1e-8) - camera_intrinsic[1][2]
                    camera_x = camera_intrinsic[0][0] * policy_input_points[:,1] / (policy_input_points[:,0] + 1e-8) + camera_intrinsic[0][2]
                    for j in range(camera_x.shape[0]-1):
                        try:
                            if camera_x[j] > 0 and camera_z[j] > 0 and camera_x[j+1] > 0 and camera_z[j+1] > 0:
                                trajectory_mask = cv2.line(trajectory_mask,(int(camera_x[j]),int(camera_z[j])),(int(camera_x[j+1]),int(camera_z[j+1])),color,3)
                        except:
                            pass
                        # if camera_x[j] > 0 and camera_z[j] > 0 and camera_x[j+1] > 0 and camera_z[j+1] > 0:
                        #     trajectory_mask = cv2.line(trajectory_mask,(int(camera_x[j]),int(camera_z[j])),(int(camera_x[j+1]),int(camera_z[j+1])),color,3)
                cv2.imwrite("test.png", trajectory_mask)
                vis_image = trajectory_mask.copy()
                # vis_image = (last_infos['observations']["policy"]['rgb'][i].cpu().numpy()*255).astype(np.uint8).transpose(1,2,0)
                fps_writer[i].append_data(vis_image)
                
        lengths = env.episode_length_buf[:args_cli.num_envs].detach().cpu().numpy()
        for i in range(args_cli.num_envs):
            new_n = int(lengths[i])
            if progress_bars[i].n != new_n:
                progress_bars[i].n = new_n
                progress_bars[i].refresh()
                
        for i in range(env.num_envs):
            if dones[i] == True and episode_num < args_cli.num_episodes:
                episode_num += 1
                if args_cli.task != "nogoal":
                    goal_poses = last_infos['observations']["policy"]['goal_pose'].cpu().numpy()[i,0:2]
                    success_flag = np.float32(is_success(np.sqrt(np.square(goal_poses).sum())))
                    evaluation_metrics.append({'success':success_flag,
                                        'spl': np.clip(euclidean[i] / trajectory_length[i],0,1) * success_flag,
                                        'distance':euclidean[i]})
                    goal_poses = infos['observations']["policy"]['goal_pose'].cpu().numpy()[i,0:2]
                    euclidean[i] = np.sqrt(np.square(goal_poses).sum())
                    trajectory_length[i] = 0
                    write_metrics(evaluation_metrics, os.path.join(log_dir_scene, f"metric_{args_cli.task}.csv"))
                else:
                    raise NotImplementedError
                if args_cli.save:
                    fps_writer[i].close()
                    fps_writer[i] = imageio.get_writer(os.path.join(log_dir_scene, "videos", f"traj_{episode_num:03d}.mp4"), fps=10)
                    print(f"[INFO] Saving videos to: {os.path.join(log_dir_scene, 'videos', f'traj_{episode_num:03d}.mp4')}")
                progress_bars[i].close()
                if episode_num < args_cli.num_episodes:
                    progress_bars[i] = tqdm(total=int(env.max_episode_length), desc=f"Episode {episode_num} (Env {i})", position=i, leave=False)
        
        if episode_num >= args_cli.num_episodes + args_cli.num_envs - 1:
            break
       
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
