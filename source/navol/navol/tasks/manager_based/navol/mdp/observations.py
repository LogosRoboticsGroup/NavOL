from __future__ import annotations

import cv2
import torch
from typing import TYPE_CHECKING

import rpyc

rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True
import atexit
import numpy as np
import os
import pickle
import socket
import threading
import torch.nn as nn
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
import timm
import torchvision
import torchvision.models as models
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.utils.math import quat_apply, quat_inv
from PIL import Image
from torchvision import transforms

from rsl_rl.networks import Memory, NavDP_RGBD_Backbone
from .gaussian_model import GSRenderer

HEAD_POS = [0.332, 0.0, 0.00]

@torch.jit.script
def euler_to_quaternion(euler_angles):
    cy = torch.cos(euler_angles[:, 2] * 0.5)
    sy = torch.sin(euler_angles[:, 2] * 0.5)
    cp = torch.cos(euler_angles[:, 1] * 0.5)
    sp = torch.sin(euler_angles[:, 1] * 0.5)
    cr = torch.cos(euler_angles[:, 0] * 0.5)
    sr = torch.sin(euler_angles[:, 0] * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack((w, x, y, z), dim=-1)


class GSServer:
    def __init__(self, host="localhost", port=12346, channels=4, time_out=10):
        self.host = host
        self.port = port
        self.channels = channels
        self.time_out = time_out
        # self.data = None
        # self.last_data = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

    def init_data(self, env_num):
        self.data = np.zeros((env_num, self.channels * 180 * 320))
        self.last_data = np.zeros((env_num, self.channels * 180 * 320))
        self.latency = np.random.randint(0, 2, size=(env_num, 1))
        self.env_num = env_num

    def receive_data(self, host="localhost", port=12345):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        conn, addr = s.accept()
        conn.settimeout(self.time_out)  # Set timeout for receiving data
        data = b""
        try:
            while True:
                packet = conn.recv(40960000)
                if not packet:
                    break
                data += packet
        except socket.timeout:
            print("No new tensor received for 10 seconds, terminating connection.")
        finally:
            conn.close()
        pickle_data = pickle.loads(data)
        return pickle_data

    def start(self):
        atexit.register(self.close)
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def close(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def run(self):
        while self.running:
            data = self.receive_data(self.host, self.port)
            with self.lock:
                self.last_data = self.data
                self.data = data

    def get_data(self):
        with self.lock:
            is_start = (self.last_data == 0).all(axis=1).reshape(-1, 1)
            latency = self.latency
            return (latency * self.last_data + (1 - latency) * self.data) * (1 - is_start) + self.data * is_start

    def reset(self, env_ids):
        if env_ids is None:
            return
        else:
            env_ids = env_ids.cpu().numpy()
            self.data[env_ids] = np.zeros((len(env_ids), self.channels * 180 * 320))
            self.last_data[env_ids] = np.zeros((len(env_ids), self.channels * 180 * 320))
            self.latency[env_ids] = np.random.randint(0, 2, size=(len(env_ids), 1))


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

import random
import threading


def base_lin_vel_zero(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.zeros(env.num_envs, 3, device=env.device)


def base_pos_z_e(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_e = asset.data.root_pos_w - env.scene.env_origins
    return pos_e[:, 2].unsqueeze(-1)


def standing_velocity_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    commands = env.command_manager.get_command(command_name)
    commands[:, :2] *= (torch.norm(commands[:, :2], dim=1) > 0.1).unsqueeze(1)
    commands[:, 2] *= torch.abs(commands[:, 2]) > 0.1
    return commands


def zero_velocity_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    commands = torch.zeros(env.num_envs, 3, device=env.device)
    return commands


def goal_pos(
    env: ManagerBasedEnv, base_height, command_name, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The goal position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    rgb_commands = env.command_manager.get_command(command_name)

    robot_pos = env.scene["robot"].data.root_pos_w
    robot_quat = env.scene["robot"].data.root_quat_w
    robot_quat_inv = quat_inv(robot_quat)

    goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - robot_pos
    goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - robot_pos
    goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - robot_pos
    goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
    
    goal_tensor = quat_apply(robot_quat_inv[:, None], goal_tensor)

    # goal_red = quat_apply(robot_quat_inv, goal_red)
    # goal_green = quat_apply(robot_quat_inv, goal_green)
    # goal_blue = quat_apply(robot_quat_inv, goal_blue)
    # goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
    goal_tensor[:, :, 2]  = 1.0
    pos_goal = (goal_tensor * rgb_commands.unsqueeze(-1)).sum(1)

    return pos_goal


def goal_pos_multi(
    env: ManagerBasedEnv, base_height, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The goal position in the environment frame."""
    goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
    # goal_tensor[:, :, 2] += base_height
    goal_tensor[:, :, 2]  = 1.0
    pos_goal = goal_tensor.reshape(env.num_envs, -1)

    return pos_goal


def head_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos_r = asset.data.root_pos_w - env.scene.env_origins
    quat_r = asset.data.root_quat_w
    head_pos = torch.tensor(HEAD_POS, device=env.device).repeat(env.num_envs, 1)
    head_pos_r = pos_r + math_utils.quat_apply(quat_r, head_pos)
    return head_pos_r


class gs_image_feature(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        self.encoder_model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
        self.encoder_model.head = nn.Identity()  # Remove the final fully connected layer
        self.encoder_model.to(env.device)
        self.encoder_model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0))], p=0.8),
                transforms.RandomApply([transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.1)], p=0.1),
                transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000]),
            ]
        )
        self.camera_pos_noise_scale = torch.tensor([0.01, 0.01, 0.01], device=env.device)
        self.camera_rot_noise_scale = torch.tensor([1.0, 1.0, 2.0], device=env.device)
        self.save_count = 0
        
        self.gaussian_renderer = GSRenderer(cfg.data_dir)
        print("GS Renderer Initialized")
        
        self.camera_pos = torch.tensor(cfg.camera_pos, device=env.device).repeat(env.num_envs, 1)
        self.camera_pos += (2 * torch.rand_like(self.camera_pos) - 1) * self.camera_pos_noise_scale
        self.camera_rot = torch.tensor(cfg.camera_rot, device=env.device).repeat(env.num_envs, 1)
        self.camera_rot += (2 * torch.rand_like(self.camera_rot) - 1) * self.camera_rot_noise_scale
        self.camera_rot = torch.deg2rad(self.camera_rot)
        self.camera_rot = euler_to_quaternion(self.camera_rot)

        self.asset_offset_pos = torch.tensor(cfg.asset_offset_pos, device=env.device).repeat(env.num_envs, 1)
        self.asset_offset_rot = torch.tensor(cfg.asset_offset_rot, device=env.device).repeat(env.num_envs, 1)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        pos_r = asset.data.root_pos_w - env.scene.env_origins
        quat_r = asset.data.root_quat_w

        camera_pos_r = pos_r + math_utils.quat_apply(quat_r, self.camera_pos) - self.asset_offset_pos
        camera_rot_r = math_utils.quat_mul(quat_r, self.camera_rot)
        camera_rot_r = math_utils.convert_camera_frame_orientation_convention(camera_rot_r, "world", "ros")
        red_cone = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins - self.asset_offset_pos
        green_cone = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins - self.asset_offset_pos
        blue_cone = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins - self.asset_offset_pos

        images = self.gaussian_renderer.render(camera_pos_r, camera_rot_r, red_cone, green_cone, blue_cone)
        
        images = self.preprocess(images)
        with torch.inference_mode():
            image_feature = self.encoder_model(images)

        return image_feature


class gs_image_feature_dp(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        self.conn = rpyc.connect("localhost", 18862)
        self.image_server = GSServer(channels=4)
        self.image_server.start()
        self.image_server.init_data(env.num_envs)
        self.image_size = 224
        self.token_dim = 384
        self.memory_size = 8

        self.rgb_token = torch.zeros(env.num_envs, 1, device=env.device)
        self.rgbd_encoder = NavDP_RGBD_Backbone(self.image_size, self.token_dim, memory_size=self.memory_size).to(self._env.device)
        # load checkpoint
        model_path = "ckpt/navdp-weights.ckpt"
        ckpt = torch.load(model_path, map_location=env.device)
        ckpt_new = {}
        for k, v in ckpt.items():
            if k.startswith("rgbd_encoder."):
                ckpt_new[k.replace("rgbd_encoder.", "")] = v

        self.rgbd_encoder.load_state_dict(ckpt_new)
        self.memory_queue = [[] for i in range(env.num_envs)]

        self.preprocess = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0))], p=0.8),
                transforms.RandomApply([transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.1)], p=0.1),
                transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000]),
            ]
        )
        self.camera_pos_noise_scale = torch.tensor([0.01, 0.01, 0.01], device=env.device)
        self.camera_rot_noise_scale = torch.tensor([1.0, 1.0, 2.0], device=env.device)
        # print("GS Server Initialized")
        self.save_count = 0

    def _process_rgb(self, rgb_images: torch.tensor) -> torch.Tensor:
        """Process RGB images with memory queue and extract tokens."""
        prop = self.image_size / 320  # max(H, W)
        H, W = rgb_images.shape[2:4]
        prop = self.image_size / max(H, W)
        rgb_images = rgb_images[:, [2, 1, 0]]
        rgb_resized = F.interpolate(rgb_images, size=(int(H * prop + 0.5), int(W * prop + 0.5)), mode='bilinear', align_corners=False)
        pad_width = max((self.image_size - rgb_resized.shape[3]) // 2, 0)
        pad_height = max((self.image_size - rgb_resized.shape[2]) // 2, 0)
        rgb_padded = F.pad(rgb_resized, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0.0)
        final_img = F.interpolate(rgb_padded, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return final_img

    def _process_depth(self, depth_images: torch.Tensor) -> torch.Tensor:
        """Process depth images and extract tokens."""
        prop = self.image_size / 320  # max(H, W)
        H, W = depth_images.shape[2:4]
        prop = self.image_size / max(H, W)
        depth_resized = F.interpolate(depth_images, size=(int(H * prop + 0.5), int(W * prop + 0.5)), mode='bilinear', align_corners=False)
        pad_width = max((self.image_size - depth_resized.shape[3]) // 2, 0)
        pad_height = max((self.image_size - depth_resized.shape[2]) // 2, 0)
        depth_padded = F.pad(depth_resized, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0.0)
        final_depth = F.interpolate(depth_padded, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        final_depth[(final_depth > 5.0) | (final_depth < 0.1)] = 0
        return final_depth

    def reset(self, env_ids: torch.Tensor | None = None):
        self.image_server.reset(env_ids)
        self.rgbd_encoder.reset(env_ids)

    def get_img_token(self) -> torch.Tensor:
        return self.rgb_token.reshape(self.num_envs, -1)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        camera_pos: list = None,
        camera_rot: list = None,
        asset_offset_pos: list = None,
        asset_offset_rot: list = None,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        pos_r = asset.data.root_pos_w - env.scene.env_origins
        quat_r = asset.data.root_quat_w
        camera_pos = torch.tensor(camera_pos, device=env.device).repeat(env.num_envs, 1)
        camera_pos += (2 * torch.rand_like(camera_pos) - 1) * self.camera_pos_noise_scale
        camera_rot = torch.tensor(camera_rot, device=env.device).repeat(env.num_envs, 1)
        camera_rot += (2 * torch.rand_like(camera_rot) - 1) * self.camera_rot_noise_scale
        camera_rot = torch.deg2rad(camera_rot)
        camera_rot = euler_to_quaternion(camera_rot)

        asset_offset_pos = torch.tensor(asset_offset_pos, device=env.device).repeat(env.num_envs, 1)
        asset_offset_rot = torch.tensor(asset_offset_rot, device=env.device).repeat(env.num_envs, 1)
        camera_pos_r = pos_r + math_utils.quat_apply(quat_r, camera_pos) - asset_offset_pos
        camera_rot_r = math_utils.quat_mul(quat_r, camera_rot)
        camera_rot_r = math_utils.convert_camera_frame_orientation_convention(camera_rot_r, "world", "ros")
        # TODO: add asset_offset_rot
        red_cone = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins - asset_offset_pos
        green_cone = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins - asset_offset_pos
        blue_cone = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins - asset_offset_pos

        camera_pos_cpu = pickle.dumps(camera_pos_r)
        camera_rot_cpu = pickle.dumps(camera_rot_r)
        red_cone_cpu = pickle.dumps(red_cone)
        green_cone_cpu = pickle.dumps(green_cone)
        blue_cone_cpu = pickle.dumps(blue_cone)

        # 使用CPU数据进行远程调用
        self.conn.root.render(camera_pos_cpu, camera_rot_cpu, red_cone_cpu, green_cone_cpu, blue_cone_cpu)
        
        image_data = torch.from_numpy(self.image_server.get_data().reshape(env.num_envs, 4, 180, 320)).float().to(env.device)

        images = self._process_rgb(image_data[:, :3])
        depths = self._process_depth(image_data[:, 3:])

        tokens = self.rgbd_encoder.get_rgbd_token(images, depths)
        return tokens.reshape(env.num_envs, -1)


def rgb_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    commands = env.command_manager.get_command(command_name)
    return commands
