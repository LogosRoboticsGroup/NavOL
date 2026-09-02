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
    "--count", type=int, default=100)
parser.add_argument(
    "--min-keypoints", type=int, default=5,
    help="Minimum number of raw shortest-path keypoints (canonical: 5).")
parser.add_argument(
    "--relax-min-keypoints", action="store_true",
    help="Legacy mode: lower the keypoint threshold after repeated failures.")
parser.add_argument(
    "--save", action='store_true', help="Whether to save the map and points.")
parser.add_argument(
    "--save_path", type=str, default="debug", help="Path to save the output files.")
parser.add_argument(
    "--npy_save_path", type=str, default="debug", help="Path to save the output files.")
args_cli = parser.parse_args()

sim = init_navmesh(args_cli.scene_path)

height = sim.scene_aabb.y().min
top_down_map = maps.get_topdown_map(
    sim.pathfinder, height, meters_per_pixel=0.05
)
recolor_map = np.array(
    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
)
top_down_map = recolor_map[top_down_map]
grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
if args_cli.save:
    os.makedirs(args_cli.save_path, exist_ok=True)
    cv2.imwrite(os.path.join(args_cli.save_path, "top_down_scene.png"), top_down_map[..., [2, 1, 0]])

start_points = []
samples = []
count = 0
seed = 0

while count < args_cli.count:
    seed += 1
    sim.pathfinder.seed(seed)
    result = navmesh_generate_goal(
        sim,
        min_keypoints=args_cli.min_keypoints,
        relax_min_keypoints=args_cli.relax_min_keypoints,
    )
    trajectory = [
        maps.to_grid(
            path_point[2],
            path_point[0],
            grid_dimensions,
            pathfinder=sim.pathfinder,
        )
        for path_point in result["path_points_nav"]
    ]
    print("Generated trajectory length:", len(trajectory))
    initial_angle = result["initial_angle"]
    samples.append(result["sample"])
    
    print(f"Trajectory {count}:  Start={result['start_point_sim']} Goal={result['end_point_sim']} Initial_Angle={initial_angle}")

    if args_cli.save:
        top_down_map_copy = top_down_map.copy()
        top_down_map_copy2 = top_down_map.copy()
        maps.draw_path(top_down_map_copy2, trajectory, thickness=5)
        maps.draw_agent(
            top_down_map_copy, trajectory[0], initial_angle+math.pi/2, agent_radius_px=24
        )
        maps.draw_agent(
            top_down_map_copy2, trajectory[0], initial_angle+math.pi/2, agent_radius_px=24
        )
        cv2.imwrite(os.path.join(args_cli.save_path, f"top_down_scene_{count:03d}.png"), top_down_map_copy[..., [2, 1, 0]])
        cv2.imwrite(os.path.join(args_cli.save_path, f"top_down_scene_{count:03d}_2.png"), top_down_map_copy2[..., [2, 1, 0]])
    count += 1

np.save(args_cli.npy_save_path, np.stack(samples))
