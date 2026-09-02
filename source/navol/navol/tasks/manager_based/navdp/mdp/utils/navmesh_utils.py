import os
import math
import cv2
import trimesh
import numpy as np
import torch
from matplotlib import pyplot as plt
import habitat_sim
from habitat.utils.visualizations import maps
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from path_smoothing import get_smooth_points



def make_simple_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def compute_navmesh(sim, height=None, radius=0.2):
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.cell_size = 0.02
    navmesh_settings.cell_height = 0.04
    if height is not None:
        navmesh_settings.agent_height = height
    else:
        navmesh_settings.agent_height = 1.0
    navmesh_settings.agent_radius = radius
    navmesh_settings.agent_max_climb = 0.002
    navmesh_settings.agent_max_slope = 15
    navmesh_settings.filter_low_hanging_obstacles = False
    navmesh_settings.filter_ledge_spans = False
    navmesh_settings.filter_walkable_low_height_spans = False
    navmesh_settings.edge_max_len = navmesh_settings.agent_radius * 2.0
    print("Computing navmesh with agent_height:", navmesh_settings.agent_height, " agent_radius:", navmesh_settings.agent_radius)
    
    navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    if navmesh_success:
        pass
        # print("Successfully recomputed the navmesh")
    else:
        raise Exception("Failed to compute the navmesh")
    # print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
    return sim
    
def init_navmesh(mesh_path, radius=0.2):
    sim_settings = {
        "scene": mesh_path,  # Scene path
        "default_agent": 0,  # Index of the default agent
        "sensor_height": 0.5,  # Height of sensors in meters, relative to the agent
        "width": 256,  # Spatial resolution of the observations
        "height": 256,
    }

    sim = habitat_sim.Simulator(make_simple_cfg(sim_settings))

    sim = compute_navmesh(sim, radius=radius)
    return sim

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

def navmesh_find_path(sim, start_pos, end_pos, sample_num=24, idx=None, start_point=None, end_point=None, search_radius=0.3):
    path = habitat_sim.ShortestPath()
    path.requested_start = start_pos
    path.requested_end = end_pos

    found_path = sim.pathfinder.find_path(path)
    geodesic_distance = path.geodesic_distance
    path.points.extend(path.requested_end)
    
    if len(path.points) < 2:
        sample_points = np.array([[0.,0.,0.],[0.,0.,0.]])
        print("Warning: No path found between start and end points.", idx, start_pos, end_pos, start_point)
        height = sim.scene_aabb.y().min
        # breakpoint()
        top_down_map = maps.get_topdown_map(
            sim.pathfinder, height, meters_per_pixel=0.1
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]
        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        path_points_nav = np.stack([start_pos, end_pos, start_point], axis=0)
        trajectory = [
            maps.to_grid(
                path_point[2],
                path_point[0],
                grid_dimensions,
                pathfinder=sim.pathfinder,
            )
            for path_point in path_points_nav
        ]
        maps.draw_path(top_down_map, trajectory)
        maps.draw_agent(
            top_down_map, trajectory[0], 0.0, agent_radius_px=8
        )
        path_points_nav = np.stack([start_point, end_point], axis=0)
        trajectory = [
            maps.to_grid(
                path_point[2],
                path_point[0],
                grid_dimensions,
                pathfinder=sim.pathfinder,
            )
            for path_point in path_points_nav
        ]
        maps.draw_path(top_down_map, trajectory[:2])
        maps.draw_agent(
            top_down_map, trajectory[0], 0.0, agent_radius_px=4
        )
        cv2.imwrite("test.png", top_down_map[..., [2, 1, 0]])

        is_valid = False
    else:
        sample_points = np.array(path.points)
        is_valid = True
    if search_radius > 0:
        # sample_points = np.stack([point if i == 0 or i == sample_points.shape[0]-1 else find_best_point_in_local_area(sim, point, search_radius=search_radius) for i, point in enumerate(sample_points)], axis=0)
        sample_points = np.stack([find_best_point_in_local_area(sim, point, search_radius=search_radius) for i, point in enumerate(sample_points)], axis=0)
    sample_points = get_smooth_points(sample_points, path_finder=sim.pathfinder, num_samples=sample_num, delta=0.25)

    distances = np.array([sim.pathfinder.distance_to_closest_obstacle(sample_points[i], 10) for i in range(sample_num)], dtype=np.float32)
    return sample_points, distances, is_valid

def find_best_point_in_local_area(sim, sample_point, search_radius=0.3, n_sample=10):
    hit_record = sim.pathfinder.closest_obstacle_surface_point(sample_point, 0.5)
    hit_normal = np.array(hit_record.hit_normal)
    hit_point = np.array(hit_record.hit_pos)
    if np.sum(hit_normal) == 0:
        return sample_point
    if np.isnan(hit_normal).any():
        rand_theta = np.random.rand(n_sample, 1) * 2 * math.pi
        rand_radius = np.random.rand(n_sample, 1) * search_radius
        rand_offset = np.concatenate([rand_radius * np.cos(rand_theta), np.zeros_like(rand_radius), rand_radius * np.sin(rand_theta)], axis=-1)
        points = sample_point[None].repeat(n_sample, axis=0) + rand_offset
    else:
        points = sample_point[None].repeat(n_sample, axis=0) + np.arange(n_sample)[:, None] * search_radius / n_sample * hit_normal
    max_distance = sim.pathfinder.distance_to_closest_obstacle(sample_point, 10)
    max_idx = -1
    for i in range(n_sample):
        if not sim.pathfinder.is_navigable(points[i]):
            continue
        dist = sim.pathfinder.distance_to_closest_obstacle(points[i], 10)
        if dist > max_distance:
            max_distance = dist
            max_idx = i
    if max_idx >= 0:
        sample_point = points[max_idx]
    return sample_point

def height_in_range(height, height_ranges):
    for height_range in height_ranges:
        if height >= height_range[0] and height <= height_range[1]:
            return True
    return False

def navmesh_generate_goal(sim, geodesic_distance_min_threshold=10.0, geodesic_distance_max_threshold=99.0, max_try=99999, height_delta=[(-0.2, 0.5)], sim_id=0, min_keypoints=5, relax_min_keypoints=False, try_keypoints_interval=10):
    count_try = 0
    height = sim.scene_aabb.y().min
    height_ranges = [(height + hr[0], height + hr[1]) for hr in height_delta]
    try_keypoints = 0
    while True:
        if count_try >  max_try:
            raise RuntimeError("Cannot find valid start and goal points in navmesh after {} tries".format(max_try), "Sim id:", sim_id)
        count_try += 1
        sample1 = np.array(sim.pathfinder.get_random_navigable_point())
        while not height_in_range(sample1[1], height_ranges):
            sample1 = np.array(sim.pathfinder.get_random_navigable_point())
        sample2 = np.array(sim.pathfinder.get_random_navigable_point())
        while not height_in_range(sample2[1], height_ranges):
            sample2 = np.array(sim.pathfinder.get_random_navigable_point())

        path = habitat_sim.ShortestPath()
        path.requested_start = sample1
        path.requested_end = sample2
        found_path = sim.pathfinder.find_path(path)
        geodesic_distance = path.geodesic_distance
        path_points = np.array(path.points)
        
        # if len(path.points) >= max_keypoints:
        #     continue
        
        if len(path.points) < min_keypoints:
            try_keypoints += 1
            if relax_min_keypoints and try_keypoints == try_keypoints_interval:
                # print("Warning: Cannot find enough keypoints in navmesh, min_keypoints:", min_keypoints, "Reducing min_keypoints.")
                try_keypoints = 0
                min_keypoints = max(2, min_keypoints - 1)
            continue
        
        if count_try % 10 == 0:
            geodesic_distance_min_threshold = max(2.0, geodesic_distance_min_threshold - 1.0)
        
        if found_path and geodesic_distance > geodesic_distance_min_threshold and geodesic_distance < geodesic_distance_max_threshold:
            sample1_sim = sample1.copy()
            sample1_sim = sample1_sim[[0, 2, 1]]
            sample1_sim[1] *= -1
            
            sample2_sim = sample2.copy()
            sample2_sim = sample2_sim[[0, 2, 1]]
            sample2_sim[1] *= -1
            
            path_points_sim = path_points.copy()
            path_points_sim = path_points_sim[:, [0, 2, 1]]
            path_points_sim[:, 1] *= -1
            
            delta = path_points_sim[1, :2] - path_points_sim[0, :2]
            initial_angle = math.atan2(delta[1], delta[0])
            
            if not np.isfinite(initial_angle):
                continue
            
            sample = np.concatenate([sample1_sim, sample2_sim, [initial_angle]], axis=-1)
            break
        else:
            continue
    result_dict = {
        "sample": sample,
        "path_points_nav": path_points,
        "path_points_sim": path_points_sim,
        "start_point_nav": sample1,
        "start_point_sim": sample1_sim,
        "end_point_nav": sample2,
        "end_point_sim": sample2_sim,
        "initial_angle": initial_angle,
    }
    return result_dict
