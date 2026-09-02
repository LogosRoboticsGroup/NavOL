import os
import math
import cv2
from matplotlib import pyplot as plt
import trimesh
import numpy as np
import habitat_sim
from habitat.utils.visualizations import maps
import magnum as mn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--scene_path", type=str, default="source/scene_data_nav/scenes/livingroom/raw_mesh/z_3d_front_livingroom_nav.obj")
parser.add_argument(
    "--count", type=int, default=100)
parser.add_argument(
    "--save", action='store_true', help="Whether to save the map and points.")
parser.add_argument(
    "--save_path", type=str, default="debug", help="Path to save the output files.")
args_cli = parser.parse_args()

def make_simple_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)

sim_settings = {
    "scene": args_cli.scene_path,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 0.5,  # Height of sensors in meters, relative to the agent
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
}
sim = habitat_sim.Simulator(make_simple_cfg(sim_settings))

navmesh_settings = habitat_sim.NavMeshSettings()
navmesh_settings.cell_size = 0.01
navmesh_settings.cell_height = 0.02
navmesh_settings.agent_height = 1.0 #0.02  #0.3
navmesh_settings.agent_radius = 0.040 * 10 #0.01  #0.4
navmesh_settings.agent_max_climb = 0.002
navmesh_settings.agent_max_slope = 15
navmesh_settings.filter_low_hanging_obstacles = False
navmesh_settings.filter_ledge_spans = False
navmesh_settings.filter_walkable_low_height_spans = False
navmesh_settings.edge_max_len = navmesh_settings.agent_radius * 2.0

meters_per_pixel = 0.1
height = sim.scene_aabb.y().min
navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
start_point = sim.pathfinder.get_random_navigable_point()
top_down_map = maps.get_topdown_map(
    sim.pathfinder, height, meters_per_pixel=meters_per_pixel
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
    sample1 = np.array(sim.pathfinder.get_random_navigable_point())
    sample2 = np.array(sim.pathfinder.get_random_navigable_point())

    if len(start_points) == 0:
        pass
    elif len(start_points) > 0 and any([np.abs(s-sample1).sum() < 1e-1 for s in start_points]):
        continue
    start_points.append(sample1)
        
    # if sample1[1] > 0 or sample2[1] > 0:
    #     continue

    path = habitat_sim.ShortestPath()
    path.requested_start = sample1
    path.requested_end = sample2
    found_path = sim.pathfinder.find_path(path)
    geodesic_distance = path.geodesic_distance
    path_points = np.array(path.points)
    print(f"Trajectory {count}:  found_path={found_path}")

    if found_path and geodesic_distance > 0.5:
        print(f"Trajectory {count}:  geodesic_distance={geodesic_distance}")
        print(f"Trajectory {count}:  path_points {len(path_points)} points")
        
        sample1_sim = sample1[[2, 0, 1]]
        sample2_sim = sample2[[2, 0, 1]]
        path_points_sim = path_points[:, [2, 0, 1]]
        trajectory = [
            maps.to_grid(
                path_point[0],
                path_point[1],
                grid_dimensions,
                pathfinder=sim.pathfinder,
            )
            for path_point in path_points_sim
        ]

        delta = path_points_sim[1, :2] - path_points_sim[0, :2]
        initial_angle = math.atan2(delta[1], delta[0])
        if not np.isfinite(initial_angle):
            continue
        
        print(f"Trajectory {count}:  Start={sample1_sim}")
        print(f"Trajectory {count}:  Goal={sample2_sim}")
        print(f"Trajectory {count}:  Angle={initial_angle}")
        print(f"Trajectory {count}:  Seed={seed}")

        top_down_map_copy = top_down_map.copy()
        maps.draw_path(top_down_map_copy, trajectory)
        maps.draw_agent(
            top_down_map_copy, trajectory[0], initial_angle, agent_radius_px=8
        )
        print("\nDisplay the map with agent and path overlay:")
        if args_cli.save:
            cv2.imwrite(os.path.join(args_cli.save_path, f"top_down_scene_{count:03d}.png"), top_down_map_copy[..., [2, 1, 0]])
        a = np.concatenate([sample1_sim, sample2_sim, [initial_angle]], axis=-1)
        samples.append(a)
        count += 1

    print("==" * 20)

save_path = os.path.join(os.path.dirname(os.path.dirname(args_cli.scene_path)), f"pointgoal_sample_{count}.npy")
np.save(save_path, np.stack(samples))