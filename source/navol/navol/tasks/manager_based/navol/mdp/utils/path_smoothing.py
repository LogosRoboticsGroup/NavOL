import numpy as np
import cv2
# import habitat_sim
from scipy.interpolate import CubicSpline

# 均匀采样
def smooth_path_uniform_sampling(path_points, num_samples=100, delta=0.2):
    if len(path_points) < 2:
        return np.array(path_points)
    
    path_array = np.array(path_points)
    
    distances = np.sqrt(np.sum(np.diff(path_array, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative_distances[-1]
    
    if total_length == 0:
        return path_array
    
    # uniform_distances = np.linspace(0, total_length, num_samples)
    uniform_distances = np.arange(0, total_length, max((total_length / (num_samples - 1)), delta))
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
    if len(path_points) < 3:
        return np.array(path_points)
    
    path_array = np.array(path_points)

    distances = np.sqrt(np.sum(np.diff(path_array, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    
    if cumulative_distances[-1] == 0:
        return path_array

    t = cumulative_distances 
    splines = []
    for dim in range(path_array.shape[1]):
        spline = CubicSpline(t, path_array[:, dim], bc_type='not-a-knot')
        splines.append(spline)
    
    # t_new = np.linspace(t[0], t[-1], num_samples)
    t_new = np.arange(t[0], t[-1], max((t[-1] - t[0]) / (num_samples - 1), delta))
    num_samples = len(t_new)
    smooth_points = np.zeros((num_samples, path_array.shape[1]))
    
    for dim, spline in enumerate(splines):
        smooth_points[:, dim] = spline(t_new)
    
    return smooth_points

def get_smooth_points(path_points, path_finder, num_samples=20):
    uniform_sampled_points = smooth_path_uniform_sampling(path_points, num_samples//2)
    cubic_spline_points = smooth_path_cubic_spline_2d(uniform_sampled_points, num_samples)
    for i in range(len(cubic_spline_points)):
        if not path_finder.is_navigable(cubic_spline_points[i]):
            cubic_spline_points[i] = path_finder.snap_point(cubic_spline_points[i])
    
    return cubic_spline_points