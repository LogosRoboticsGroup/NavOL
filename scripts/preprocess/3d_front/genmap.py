import os
import math
import cv2
import trimesh
import numpy as np
import habitat_sim
from habitat.utils.visualizations import maps
import magnum as mn
import argparse

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "../source/navol/navol/tasks/manager_based/navdp/mdp/utils/"))
print(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "../source/navol/navol/tasks/manager_based/navdp/mdp/utils/"))
from navmesh_utils import init_navmesh, navmesh_generate_goal

parser = argparse.ArgumentParser()
parser.add_argument(
    "--scene_path", type=str, default="source/scene_data_nav/scenes/livingroom/raw_mesh/z_3d_front_livingroom_nav.obj")
parser.add_argument(
    "--save_path", type=str, default="source/scene_data_3d_front/3d_front_scene_1000/maps", help="Path to save the output files.")
args_cli = parser.parse_args()

sim = init_navmesh(args_cli.scene_path)

height = sim.scene_aabb.y().min
top_down_map = maps.get_topdown_map(
    sim.pathfinder, height, meters_per_pixel=0.1
)
recolor_map = np.array(
    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
)
top_down_map = recolor_map[top_down_map]
grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
os.makedirs(args_cli.save_path, exist_ok=True)
cv2.imwrite(os.path.join(args_cli.save_path, os.path.basename(args_cli.scene_path[:-4]+'.png')), top_down_map[..., [2, 1, 0]])
print(f"Saved top-down map to {os.path.join(args_cli.save_path, os.path.basename(args_cli.scene_path[:-4]+'.png'))}")