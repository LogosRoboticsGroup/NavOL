#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import numpy as np
import os
import torch
from torch import nn
import json
import einops
from einops import einsum
from e3nn import o3

from plyfile import PlyData
from .gaussian_utils import quaternion_to_matrix, matrix_to_quaternion, build_scaling_rotation, \
    inverse_sigmoid, strip_symmetric, getProjectionMatrix, fov2focal, Timing
# from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
# from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = torch.nn.functional.pad(bin_img, pad=[pad, pad, pad, pad], mode="reflect")
    out = torch.nn.functional.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out


def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.Fx = fov2focal(self.FoVx, self.image_width)
        self.Fy = fov2focal(self.FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor(
            [[self.Fx / scale, 0, self.Cx / scale], [0, self.Fy / scale, self.Cy / scale], [0, 0, 1]]
        ).float()
        extrinsic_matrix = self.world_view_transform.transpose(0, 1).contiguous()  # cam2world
        return intrinsic_matrix, extrinsic_matrix


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._knn_f = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.max_weight = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.denom_abs = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.knn_dists = None
        self.knn_idx = None
        self.setup_functions()
        self.use_app = False
        
        self.cov3D_precomp = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._knn_f,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.max_weight,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.denom_abs,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._knn_f,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.max_weight,
            xyz_gradient_accum,
            xyz_gradient_accum_abs,
            denom,
            denom_abs,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.denom_abs = denom_abs
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)

    def get_normal(self, view_cam):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global

    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)

    def get_covariance(self, scaling_modifier=1):
        if self.cov3D_precomp is None:
            self.cov3D_precomp = self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)
        return self.cov3D_precomp

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append(f"f_dc_{i}")
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append(f"f_rest_{i}")
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            l.append(f"rot_{i}")
        return l

    def load_ply(self, path, device='cuda'):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree



def transform_gaussians(gaussians, T, scale: float):
    """
    Apply an in-place 4x4 similarity transform (rotation + translation + uniform scale)
    to a set of Gaussian primitives.
    """
    device = gaussians._xyz.device

    # 1. Update Gaussian centers -----------------------------------------------
    ones = torch.ones((gaussians._xyz.shape[0], 1), device=device)
    xyz_h = torch.cat([gaussians._xyz, ones], dim=1)
    transformed_xyz_h = torch.matmul(torch.tensor(T, device=device).float(), xyz_h.transpose(0, 1)).transpose(0, 1)
    gaussians._xyz = transformed_xyz_h[:, :3]

    # 2. Update isotropic scale (log‑space) -------------------------------------
    gaussians._scaling += np.log(scale)

    # 3. Update rotation (stored as quaternion) ---------------------------------
    rotation_norm = T[:3, :3] / scale
    rotation_matrix = gaussians.get_rotation_matrix()
    new_rotation = torch.matmul(torch.tensor(rotation_norm, device=device).float(), rotation_matrix)
    gaussians._rotation = matrix_to_quaternion(new_rotation)

    # 4. Rotate spherical harmonics (SH) coefficients ---------------------------
    shs_feat = gaussians._features_rest.cpu().double()
    shs_feat = transform_shs(shs_feat, rotation_norm)
    gaussians._features_rest = shs_feat.float().to(device)

    return gaussians

@torch.jit.script
def rotation_matrix_from_quaternion(quaternion):
    """Convert a batch of quaternions *(w, x, y, z)* into 3x3 rotation matrices."""
    q = quaternion
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Elements of the rotation matrix (row‑major)
    R = torch.stack(
        [
            torch.stack([1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q3 * q0, 2 * q1 * q3 + 2 * q2 * q0], dim=1),
            torch.stack([2 * q1 * q2 + 2 * q3 * q0, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q1 * q0], dim=1),
            torch.stack([2 * q1 * q3 - 2 * q2 * q0, 2 * q2 * q3 + 2 * q1 * q0, 1 - 2 * q1 * q1 - 2 * q2 * q2], dim=1),
        ],
        dim=1,
    )

    return R

def to_so3(R: torch.Tensor) -> torch.Tensor:
    U, _, Vt = torch.linalg.svd(R)
    R_orth = U @ Vt

    if torch.det(R_orth) < 0:
        U[..., -1] *= -1
        R_orth = U @ Vt

    return R_orth

def transform_shs(shs_feat, rotation_matrix):
    """Rotate SH features up to order 3."""
    # switch axes: yzx -> xyz
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
    rotation_matrix_fix = to_so3(torch.from_numpy(permuted_rotation_matrix))
    rot_angles = o3._rotation.matrix_to_angles(rotation_matrix_fix)

    # Wigner‑D blocks -----------------------------------------------------------
    D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2])

    # rotation of the shs features
    ## order‑1 SH ---------------------------------------------------------------
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, "n shs_num rgb -> n rgb shs_num")
    one_degree_shs = einsum(
        D_1,
        one_degree_shs,
        "... i j, ... j -> ... i",
    )
    one_degree_shs = einops.rearrange(one_degree_shs, "n rgb shs_num -> n shs_num rgb")
    shs_feat[:, 0:3] = one_degree_shs

    ## order‑2 SH ---------------------------------------------------------------
    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, "n shs_num rgb -> n rgb shs_num")
    two_degree_shs = einsum(
        D_2,
        two_degree_shs,
        "... i j, ... j -> ... i",
    )
    two_degree_shs = einops.rearrange(two_degree_shs, "n rgb shs_num -> n shs_num rgb")
    shs_feat[:, 3:8] = two_degree_shs

    ## order‑3 SH ---------------------------------------------------------------
    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, "n shs_num rgb -> n rgb shs_num")
    three_degree_shs = einsum(
        D_3,
        three_degree_shs,
        "... i j, ... j -> ... i",
    )
    three_degree_shs = einops.rearrange(three_degree_shs, "n rgb shs_num -> n shs_num rgb")
    shs_feat[:, 8:15] = three_degree_shs

    return shs_feat


def filter_gaussians_within_bounding_box(gaussians, bounding_box, width):
    """
    Keep only those Gaussians whose centres fall inside a *local* axis-aligned
    box after transforming them into the box's coordinate frame.

    The operation is **in-place**: all per-Gaussian attributes are masked.
    """
    device = gaussians._xyz.device
    bounding_box = np.linalg.inv(bounding_box)

    # Homogeneous coordinates of Gaussian centres
    ones = torch.ones((gaussians._xyz.shape[0], 1), device=device)
    xyz_h = torch.cat([gaussians._xyz, ones], dim=1)
    transformed_xyz_h = torch.matmul(
        torch.tensor(bounding_box, device=device).float(), xyz_h.transpose(0, 1)
    ).transpose(0, 1)
    transformed_xyz = transformed_xyz_h[:, :3]

    mask = (
        (transformed_xyz[:, 0] >= -width[0] / 2)
        & (transformed_xyz[:, 0] <= width[0] / 2)
        & (transformed_xyz[:, 1] >= -width[1] / 2)
        & (transformed_xyz[:, 1] <= width[1] / 2)
        & (transformed_xyz[:, 2] >= -width[2] / 2)
        & (transformed_xyz[:, 2] <= width[2] / 2)
    )

    # Apply mask to *all* per‑Gaussian attributes
    for attr in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
        if hasattr(gaussians, attr):
            setattr(gaussians, attr, getattr(gaussians, attr)[mask])

    return gaussians

class GSRenderer:
    """Convenience wrapper that loads four *Gaussian Splatting* point-clouds
    (environment + three coloured groups) and renders RGB images from arbitrary
    camera poses in simulation environment."""

    def __init__(self, data_dir: str = "vr-robo-dataset", device='cuda'):
        with torch.inference_mode():
            self.device = device

            self.gaussians_env = GaussianModel(sh_degree=3)
            self.gaussians_red = GaussianModel(sh_degree=3)
            self.gaussians_green = GaussianModel(sh_degree=3)
            self.gaussians_blue = GaussianModel(sh_degree=3)

            self.gaussians_env.load_ply(f"{data_dir}/pcd/scene/point_cloud.ply", device=self.device)
            self.gaussians_red.load_ply(f"{data_dir}/pcd/red/point_cloud.ply", device=self.device)
            self.gaussians_green.load_ply(f"{data_dir}/pcd/green/point_cloud.ply", device=self.device)
            self.gaussians_blue.load_ply(f"{data_dir}/pcd/blue/point_cloud.ply", device=self.device)

            with open(f"{data_dir}/transform.json") as f:
                params = json.load(f)

            T_env = np.array(params["env"]["T"])
            scale_env = params["env"]["scale"]

            bounding_box_red = np.array(params["red"]["bounding_box"])
            width_red = np.array(params["red"]["width"])
            T_red = np.array(params["red"]["T"])
            scale_red = params["red"]["scale"]

            bounding_box_green = np.array(params["green"]["bounding_box"])
            width_green = np.array(params["green"]["width"])
            T_green = np.array(params["green"]["T"])
            scale_green = params["green"]["scale"]

            bounding_box_blue = np.array(params["blue"]["bounding_box"])
            width_blue = np.array(params["blue"]["width"])
            T_blue = np.array(params["blue"]["T"])
            scale_blue = params["blue"]["scale"]

            # ----- Filter + transform each coloured cloud --------------
            self.gaussians_red = filter_gaussians_within_bounding_box(self.gaussians_red, bounding_box_red, width_red)
            self.gaussians_green = filter_gaussians_within_bounding_box(
                self.gaussians_green, bounding_box_green, width_green
            )
            self.gaussians_blue = filter_gaussians_within_bounding_box(
                self.gaussians_blue, bounding_box_blue, width_blue
            )

            self.gaussians_env = transform_gaussians(self.gaussians_env, T_env, scale_env)
            self.gaussians_red = transform_gaussians(self.gaussians_red, T_red, scale_red)
            self.gaussians_green = transform_gaussians(self.gaussians_green, T_green, scale_green)
            self.gaussians_blue = transform_gaussians(self.gaussians_blue, T_blue, scale_blue)

            # Additional +90° around Z for green & blue (sim‑specific) ---------
            T_Z_90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            self.gaussians_green = transform_gaussians(self.gaussians_green, T_Z_90, 1.0)
            self.gaussians_blue = transform_gaussians(self.gaussians_blue, T_Z_90, 1.0)

            models = [self.gaussians_env, self.gaussians_red, self.gaussians_green, self.gaussians_blue]
            counts = [m._xyz.shape[0] for m in models]
            fields = ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]
            for field in fields:
                setattr(self.gaussians_env, field, torch.cat([getattr(m, field) for m in models], dim=0))

            self.start_indices = np.cumsum([0] + counts).tolist()[1:]

            # ----- Camera calibrition (adjust according to your own camera)-------------
            self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
            self.width, self.height = 320, 180
            self.fovx, self.fovy   = 1.5701, 1.0260
            self.znear, self.zfar  = 0.01, 100.0

    def render(
        self,
        pos: torch.Tensor,
        ori: torch.Tensor,
        red_pos: torch.Tensor,
        green_pos: torch.Tensor,
        blue_pos: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            num_poses = len(pos)
            rotation = rotation_matrix_from_quaternion(ori)

            T_sim = torch.zeros([num_poses, 4, 4], device=self.device)
            T_sim[:, :3, :3] = rotation
            T_sim[:, :3, 3] = pos
            T_sim[:, 3, 3] = 1
            T_sim = torch.inverse(T_sim)  # camera→world

            render_images = []
            # --- Iterate over views ------------------------------------
            for i in range(num_poses):
                world_view_transform = T_sim[i].transpose(0, 1)
                projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.fovx, fovY=self.fovy, device=self.device).transpose(0, 1)
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

                custom_cam = MiniCam(
                    self.width,
                    self.height,
                    self.fovy,
                    self.fovx,
                    self.znear,
                    self.zfar,
                    world_view_transform,
                    full_proj_transform,
                )
                self.gaussians_env._xyz[self.start_indices[0] : self.start_indices[1]] += red_pos[i]
                self.gaussians_env._xyz[self.start_indices[1] : self.start_indices[2]] += green_pos[i]
                self.gaussians_env._xyz[self.start_indices[2] : self.start_indices[3]] += blue_pos[i]
                render_pkg = self.rasterize(custom_cam, return_plane=False)
                self.gaussians_env._xyz[self.start_indices[0] : self.start_indices[1]] -= red_pos[i]
                self.gaussians_env._xyz[self.start_indices[1] : self.start_indices[2]] -= green_pos[i]
                self.gaussians_env._xyz[self.start_indices[2] : self.start_indices[3]] -= blue_pos[i]
                rendering = render_pkg["render"]
                render_images.append(rendering)

            render_images = torch.stack(render_images)
        return render_images
    
    @torch.no_grad()
    def rasterize(
        self,
        viewpoint_camera,
        return_plane=True,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.gaussians_env.get_xyz, dtype=self.gaussians_env.get_xyz.dtype, requires_grad=True, device=self.device) + 0
        screenspace_points_abs = torch.zeros_like(self.gaussians_env.get_xyz, dtype=self.gaussians_env.get_xyz.dtype, requires_grad=True, device=self.device) + 0
        
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        means3D = self.gaussians_env.get_xyz
        means2D = screenspace_points
        means2D_abs = screenspace_points_abs
        opacity = self.gaussians_env.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = self.gaussians_env.get_covariance(1.0)

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.

        shs = self.gaussians_env.get_features
        colors_precomp = None

        return_dict = None
        raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.background,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians_env.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            debug=False,
        )

        rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

        if not return_plane:
            rendered_image, radii, out_observe, _, _ = rasterizer(
                means3D=means3D,
                means2D=means2D,
                means2D_abs=means2D_abs,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )

            return_dict = {
                "render": rendered_image,
                "out_observe": out_observe,
            }
            return return_dict

        global_normal = self.gaussians_env.get_normal(viewpoint_camera)
        local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3]
        pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3, :3] + viewpoint_camera.world_view_transform[3, :3]
        depth_z = pts_in_cam[:, 2]
        local_distance = (local_normal * pts_in_cam).sum(-1).abs()
        input_all_map = torch.zeros((means3D.shape[0], 5), device=self.device).float()
        input_all_map[:, :3] = local_normal
        input_all_map[:, 3] = 1.0
        input_all_map[:, 4] = local_distance

        rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
            means3D=means3D,
            means2D=means2D,
            means2D_abs=means2D_abs,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            all_map=input_all_map,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_normal = out_all_map[0:3]
        rendered_alpha = out_all_map[3:4,]
        rendered_distance = out_all_map[4:5,]

        return_dict = {
            "render": rendered_image,
            "visibility_filter": radii > 0,
            "radii": radii,
            "out_observe": out_observe,
            "rendered_normal": rendered_normal,
            "plane_depth": plane_depth,
            "rendered_distance": rendered_distance,
        }

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return return_dict
