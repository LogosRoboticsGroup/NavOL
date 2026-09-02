from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from scipy.spatial.transform import Rotation as R
import rpyc
from rsl_rl.networks import Memory, NavDP_RGBD_Backbone, NavDP_ImageGoal_Backbone
import numpy as np
import cv2
rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True

import isaaclab.utils.math as math_utils
import navol.tasks.manager_based.navdp.mdp as mdp
from navol.paths import model_path as resolve_model_path
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaacsim.core.prims import XFormPrim

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def process_rgb(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera"), image_size=224) -> torch.Tensor:
    """Process RGB images with memory queue and extract tokens."""
    asset = env.scene[asset_cfg.name]
    rgb_images = asset.data.output['rgb']
    rgb_images = rgb_images.permute(0, 3, 1, 2) / 255.0  # (B,3,H,W)
    H, W = rgb_images.shape[2:4]
    prop = image_size / max(H, W)
    rgb_resized = F.interpolate(rgb_images, size=(int(H * prop + 0.5), int(W * prop + 0.5)), mode='bilinear', align_corners=False)
    pad_width = max((image_size - rgb_resized.shape[3]) // 2, 0)
    pad_height = max((image_size - rgb_resized.shape[2]) // 2, 0)
    final_img = F.pad(rgb_resized, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0.0)
    final_img = F.interpolate(final_img, size=(image_size, image_size), mode='bilinear', align_corners=False)
    return final_img

def process_depth(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera"), image_size=224) -> torch.Tensor:
    """Process depth images and extract tokens."""
    asset = env.scene[asset_cfg.name]
    depth_images = asset.data.output['distance_to_image_plane']
    depth_images = torch.where(torch.isinf(depth_images), torch.zeros_like(depth_images), depth_images)
    depth_images = depth_images.permute(0, 3, 1, 2)  # (B,1,H,W)
    H, W = depth_images.shape[2:4]
    prop = image_size / max(H, W)
    depth_resized = F.interpolate(depth_images, size=(int(H * prop + 0.5), int(W * prop + 0.5)), mode='bilinear', align_corners=False)
    pad_width = max((image_size - depth_resized.shape[3]) // 2, 0)
    pad_height = max((image_size - depth_resized.shape[2]) // 2, 0)
    depth_padded = F.pad(depth_resized, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0.0)
    final_depth = F.interpolate(depth_padded, size=(image_size, image_size), mode='bilinear', align_corners=False)
    final_depth[(final_depth > 5.0) | (final_depth < 0.1)] = 0
    return final_depth



def oracle_imu_pose_data(env: ManagerBasedEnv, 
                         robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot_asset = env.scene[robot_asset_cfg.name]
    robot_rot = math_utils.matrix_from_quat(robot_asset.data.root_quat_w)
    robot_pos = robot_asset.data.root_pos_w
    goal_primview = XFormPrim(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view")
    goal_pos = goal_primview.get_world_poses()[0]
    rel_pos = torch.matmul(torch.inverse(robot_rot), (goal_pos - robot_pos)[:, :, None]).squeeze(-1)
    pose = rel_pos[:, 0:2]
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)
    if pose.shape[-1] == 2:
        z_coords = torch.zeros((pose.shape[0], 1), device=env.device)
        pose = torch.cat([pose, z_coords], dim=1)
    return pose

def goal_pos_w(env: ManagerBasedEnv):
    goal_primview = XFormPrim(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view")
    goal_pos = goal_primview.get_world_poses()[0]
    return goal_pos

def pixel_projection_data(env: ManagerBasedEnv,
                          robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("camera")):
    camera_asset = env.scene[robot_asset_cfg.name]
    camera_w_pos = camera_asset._data.pos_w 
    camera_w_rot = math_utils.matrix_from_quat(camera_asset._data.quat_w_world)
    camera_intrinsic = camera_asset._data.intrinsic_matrices
    goal_primview = XFormPrim(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view")
    goal_pos = goal_primview.get_world_poses()[0]
    pixel_coords = torch.zeros((goal_pos.shape[0], 2))
    for i in range(camera_intrinsic.shape[0]):
        frame_coord = torch.matmul(torch.inverse(camera_w_rot[i]),(goal_pos[i] - camera_w_pos[i]).T)
        pixel_coord_x =  -frame_coord[1] * camera_intrinsic[i,0,0] / frame_coord[0] + camera_intrinsic[i,0,2]
        pixel_coord_y =  -frame_coord[2] * camera_intrinsic[i,1,1] / frame_coord[0] + camera_intrinsic[i,1,2]
        pixel_coords[i] = torch.as_tensor([pixel_coord_x,pixel_coord_y],dtype=torch.float32,device=camera_w_pos.device)
    return pixel_coords

def rgb_only(env: ManagerBasedEnv, image_size=224):
    rgb = process_rgb(env, asset_cfg=SceneEntityCfg("camera_sensor"), image_size=image_size)
    return rgb

def depth_only(env: ManagerBasedEnv, image_size=224):
    depth = process_depth(env, asset_cfg=SceneEntityCfg("camera_sensor"), image_size=image_size)
    return depth

class RGBD_feature(ManagerTermBase):
    def __init__(self, cfg: mdp.RGBDFeatureCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        self.image_size = cfg.image_size
        self.token_dim = cfg.token_dim
        self.memory_size = cfg.memory_size

        self.rgbd_encoder = NavDP_RGBD_Backbone(self.image_size, self.token_dim, memory_size=self.memory_size).to(self._env.device)
        checkpoint_path = resolve_model_path("navdp-cross-modal.ckpt")
        ckpt = torch.load(checkpoint_path, map_location=env.device, weights_only=True)
        ckpt_new = {}
        for k, v in ckpt.items():
            if k.startswith("rgbd_encoder."):
                ckpt_new[k.replace("rgbd_encoder.", "")] = v

        self.rgbd_encoder.load_state_dict(ckpt_new)
        self.rgbd_encoder.eval()
        
    def reset(self, env_ids: torch.Tensor | None = None):
        if self.rgbd_encoder.memory_queue is not None:
            self.rgbd_encoder.reset(env_ids)


    
    def process_image(self,images):
        assert len(images.shape) == 4
        H,W,C = images.shape[1],images.shape[2],images.shape[3]
        prop = self.image_size/max(H,W)
        return_images = []
        for img in images:
            resize_image = cv2.resize(img,(-1,-1),fx=prop,fy=prop)
            pad_width = max((self.image_size - resize_image.shape[1])//2,0)
            pad_height = max((self.image_size - resize_image.shape[0])//2,0)
            pad_image = np.pad(resize_image,((pad_height,pad_height),(pad_width,pad_width),(0,0)),mode='constant',constant_values=0)
            resize_image = cv2.resize(pad_image,(self.image_size,self.image_size))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            return_images.append(resize_image)
        return np.array(return_images)

    def process_depth(self,depths):
        assert len(depths.shape) == 4
        depths[depths==np.inf] = 0
        H,W,C = depths.shape[1],depths.shape[2],depths.shape[3]
        prop = self.image_size/max(H,W)
        return_depths = []
        for depth in depths:
            resize_depth = cv2.resize(depth,(-1,-1),fx=prop,fy=prop)
            pad_width = max((self.image_size - resize_depth.shape[1])//2,0)
            pad_height = max((self.image_size - resize_depth.shape[0])//2,0)
            pad_depth = np.pad(resize_depth,((pad_height,pad_height),(pad_width,pad_width)),mode='constant',constant_values=0)
            resize_depth = cv2.resize(pad_depth,(self.image_size,self.image_size))
            resize_depth[resize_depth>5.0] = 0
            resize_depth[resize_depth<0.1] = 0
            return_depths.append(resize_depth[:,:,np.newaxis])
        return np.array(return_depths)
    
    def __call__(
        self,
        env: ManagerBasedEnv,
    ) -> torch.Tensor:
        
        rgb = process_rgb(env, asset_cfg=SceneEntityCfg("camera_sensor"), image_size=self.image_size)
        depth = process_depth(env, asset_cfg=SceneEntityCfg("camera_sensor"), image_size=self.image_size)
        tokens = self.rgbd_encoder.get_rgbd_token(rgb, depth)
            
        # images = (rgb.permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)
        # depths = depth.permute(0,2,3,1).cpu().numpy()
        
        # # images = (rgb.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
        # # depths = depth.permute(0,2,3,1).cpu().numpy()
        # # process_images = rgb.permute(0, 2, 3, 1).cpu().numpy()
        # # process_depths = depth.permute(0, 2, 3, 1).cpu().numpy()
        
        
        # # asset = env.scene[SceneEntityCfg("camera_sensor").name]
        # # images = asset.data.output['rgb'].cpu().numpy().astype(np.uint8)
        # # asset = env.scene[SceneEntityCfg("camera_sensor").name]
        # # depths = asset.data.output['distance_to_image_plane'].cpu().numpy()
        # process_images = self.process_image(images)
        # depths = np.clip(depths*10000.0,0,65535.0).astype(np.uint16).astype(np.float32)/10000.0
        
        # process_depths = self.process_depth(depths)
        
        # # process_images = rgb.permute(0,2,3,1).cpu().numpy()
        # # process_depths = depths
        
        # input_images = []
        # for i in range(len(self.memory_queue2)):
        #     if len(self.memory_queue2[i]) < self.memory_size:
        #         self.memory_queue2[i].append(process_images[i])
        #         input_image = np.array(self.memory_queue2[i])
        #         input_image = np.pad(input_image,((self.memory_size - input_image.shape[0],0),(0,0),(0,0),(0,0)))
        #     else:
        #         del self.memory_queue2[i][0]
        #         self.memory_queue2[i].append(process_images[i])    
        #         input_image = np.array(self.memory_queue2[i])
                
        #     input_images.append(input_image)
        # input_image = np.array(input_images)
        # input_depth = process_depths
        # # image_tokens, tokens = self.rgbd_encoder(input_image,input_depth)
        
        # self.input_image = input_image
        # self.input_depth = input_depth
        
        # tokens = self.navi_former.rgbd_encoder(input_image,input_depth)
    
        # tokens3 = self.rgbd_encoder.get_rgbd_token(rgb, depth)
        return tokens

    
class Imagegoal_feature(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        self.image_size = 224
        self.token_dim = 384
        self.memory_size = 8
        self.memory_queue = None

        self.zero_image = torch.zeros((1, 3, 224, 224), device=self.device)
        self.image_encoder = NavDP_ImageGoal_Backbone(self.image_size,self.token_dim).to(self._env.device)
        checkpoint_path = resolve_model_path("navdp-cross-modal.ckpt")
        ckpt = torch.load(checkpoint_path, map_location=env.device, weights_only=True)
        ckpt_new = {}
        for k, v in ckpt.items():
            if k.startswith("image_encoder."):
                ckpt_new[k.replace("image_encoder.", "")] = v

        self.image_encoder.load_state_dict(ckpt_new)


    def __call__(
        self,
        env: ManagerBasedEnv,
    ) -> torch.Tensor:
        rgb = process_rgb(env, asset_cfg=SceneEntityCfg("camera_sensor"), image_size=self.image_size)
        goal_image = process_rgb(env, asset_cfg=SceneEntityCfg("goal_camera"), image_size=self.image_size)
        B = rgb.shape[0]

        if self.memory_queue is None:
            self.memory_queue = self.zero_image[:, None].repeat(B, self.memory_size, 1, 1, 1)
        self.memory_queue = torch.cat((self.memory_queue[:, 1:], rgb[:, None]), dim=1)
        
        tokens = self.image_encoder(torch.cat((goal_image,self.memory_queue[:,-1]), dim=1))
        return tokens
    
    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is not None:
            self.memory_queue[env_ids] = self.zero_image[:, None].repeat(env_ids.shape[0], self.memory_size, 1, 1, 1)
        else:
            self.memory_queue = self.zero_image[:, None].repeat(self.memory_queue.shape[0], self.memory_size, 1, 1, 1)
