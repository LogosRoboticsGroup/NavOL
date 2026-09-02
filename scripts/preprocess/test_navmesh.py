import math
import cv2
import trimesh
import numpy as np
import habitat_sim
from habitat.utils.visualizations import maps
import magnum as mn
# from navol.tasks.manager_based.navol.mdp.utils.path_smoothing import smooth_path_bezier, smooth_path_uniform_sampling
from path_smoothing import smooth_path_bezier, smooth_path_uniform_sampling

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

sim_settings = {
    # "scene": cfg.mesh_path,  # Scene path
    "scene": "ckpt/combined2.ply",  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 0.5,  # Height of sensors in meters, relative to the agent
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
}
sim = habitat_sim.Simulator(make_simple_cfg(sim_settings))

navmesh_settings = habitat_sim.NavMeshSettings()
navmesh_settings.cell_size = 0.01
navmesh_settings.cell_height = 0.02
navmesh_settings.agent_height = 0.3
navmesh_settings.agent_radius = 0.35
navmesh_settings.agent_max_climb = 0.05
navmesh_settings.agent_max_slope = 45
navmesh_settings.edge_max_len = navmesh_settings.agent_radius * 2.0

navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
top_down_map = maps.get_topdown_map(sim.pathfinder, height=sim.pathfinder.get_bounds()[0][1], meters_per_pixel=0.01)
recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
top_down_map = recolor_map[top_down_map]
grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
# convert world agent position to maps module grid point

sample1 = np.array([-0.525696, 0.4, 0.413413])
sample2 = np.array([-0.755209, 0.3556, 2.3407])

sample1 = np.array([-0.598261, 0.342396, 2.496715])
sample2 = np.array([0.985114, -0.000741, 2.221107])

path = habitat_sim.ShortestPath()
path.requested_start = sample1
path.requested_end = sample2
found_path = sim.pathfinder.find_path(path)
geodesic_distance = path.geodesic_distance
path_points = np.array(path.points)

sample_points = smooth_path_bezier(path_points, num_samples=20)
sample_points = smooth_path_uniform_sampling(sample_points, num_samples=20)
# sample_points = smooth_path_cubic_bezier(path_points, num_samples=20)

# for i in range(len(path_points)):
#     agent_grid_pos = maps.to_grid(
#         path_points[i][2], path_points[i][0], grid_dimensions, pathfinder=sim.pathfinder
#     )
#     maps.draw_agent(
#         top_down_map, agent_grid_pos, 0., agent_radius_px=8
#     )
# cv2.imwrite("top_down_map_ori.png", top_down_map[..., [2, 1, 0]])


for i in range(len(sample_points)):
    agent_grid_pos = maps.to_grid(
        sample_points[i][2], sample_points[i][0], grid_dimensions, pathfinder=sim.pathfinder
    )
    maps.draw_agent(
        top_down_map, agent_grid_pos, 0., agent_radius_px=8
    )
cv2.imwrite("top_down_map.png", top_down_map[..., [2, 1, 0]])

print(path_points)
# cv2.imwrite("test2.png", self.sim.get_sensor_observations()["color_sensor"][..., [2, 1, 0, 3]])
# mesh = trimesh.load(self.mesh_path, process=False)
# trimesh.Trimesh(np.array(path_points)).export("path.ply")