import numpy as np
import cv2
# import habitat_sim
from scipy.interpolate import CubicSpline

# 均匀采样
def smooth_path_uniform_sampling(path_points, num_samples=100, delta=0.2, random_offset=False, n_trajectory=10):
    if len(path_points) < 2:
        return np.array(path_points)
    
    path_array = np.array(path_points)
    
    distances = np.sqrt(np.sum(np.diff(path_array, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative_distances[-1]
    
    if total_length == 0:
        return path_array
    
    # uniform_distances = np.linspace(0, total_length, num_samples)
    delta = min(delta, total_length / num_samples)
    if total_length > num_samples * delta:
        uniform_distances = np.arange(0, total_length, delta)[:num_samples]
    else:
        uniform_distances = np.linspace(0, total_length, num_samples)
    if random_offset:
        uniform_distances += (2 * np.random.rand() - 1) * 0.5 * delta
    smooth_points = []
    
    for target_dist in uniform_distances:
        idx = np.searchsorted(cumulative_distances, target_dist)
        
        if idx == 0:
            smooth_points.append(path_array[0])
        elif idx >= len(path_array):
            smooth_points.append(path_array[-1])
        else:
            d0, d1 = cumulative_distances[idx-1], cumulative_distances[idx]
            p0, p1 = path_array[idx-1], path_array[idx]
            
            if d1 != d0:
                weight = (target_dist - d0) / (d1 - d0)
                point = (1 - weight) * p0 + weight * p1
            else:
                point = p0
            
            smooth_points.append(point)
    
    return np.array(smooth_points)

# Cubic spline
def smooth_path_cubic_spline_2d(path_points, num_samples=100, delta=0.2):
    t = np.linspace(0, 1, path_points.shape[0])
    cs_x = CubicSpline(t, path_points[:, 0])
    cs_y = CubicSpline(t, path_points[:, 1])
    cs_z = CubicSpline(t, path_points[:, 2])
    t_fine = np.linspace(0, 1, num_samples)
    x_fine = cs_x(t_fine)
    y_fine = cs_y(t_fine)
    z_fine = cs_z(t_fine)
    smooth_points = np.stack((x_fine, y_fine, z_fine), axis=-1)
    return smooth_points

def get_smooth_points(path_points, path_finder, num_samples=24, delta=0.25, random_offset=False, n_trajectory=10):
    uniform_sampled_points = smooth_path_uniform_sampling(path_points, num_samples//2, delta=delta, random_offset=random_offset, n_trajectory=n_trajectory)
    cubic_spline_points = smooth_path_cubic_spline_2d(uniform_sampled_points, num_samples)
    for i in range(len(cubic_spline_points)):
        if not path_finder.is_navigable(cubic_spline_points[i]):
            cubic_spline_points[i] = path_finder.snap_point(cubic_spline_points[i])
    
    return cubic_spline_points