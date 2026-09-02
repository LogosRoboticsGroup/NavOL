import os

import cv2
import sys

save_dir = "results_vis/figures_supp_traj"
os.makedirs(save_dir, exist_ok=True)
data_list = [
    [[0, 17, 29, 47], "source/scene_data_benchmark_scenes_in_domain/benchmark_in_domain/scene_000/trajectory"],
    [[4, 7, 11, 21], "source/scene_data_benchmark_scenes_in_domain/benchmark_in_domain/scene_007/trajectory"],
    [[6, 14, 29, 42], "source/scene_data_benchmark_scenes_out_domain/benchmark_out_domain/scene_001/trajectory"],
    [[0, 10, 14, 21], "source/scene_data_benchmark_scenes_out_domain/benchmark_out_domain/scene_007/trajectory"],
]
for i, (idx_list, trajectory_path) in enumerate(data_list):
    for j, idx in enumerate(idx_list):
        source_path = os.path.join(trajectory_path, f"top_down_scene_{idx:03d}_2.png")
        target_path = os.path.join(save_dir, f"{i}_{j}.png")
        cmd = f"cp {source_path} {target_path}"
        os.system(cmd)
