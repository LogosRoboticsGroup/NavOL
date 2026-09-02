import os

import cv2
import sys

save_dir = "results_vis/figures_supp_indomain"
os.makedirs(save_dir, exist_ok=True)
data_list = [
    [[9, 13, 21, 23.5, 25], "results/full_1116_step128/benchmark_in_domain_scene_002_step128_mpc_mpc_500/videos/traj_057.mp4"],
    [[1, 13, 21, 23, 26], "results/full_1116_step128/benchmark_in_domain_scene_003_step128_mpc_mpc_500/videos/traj_000.mp4"],
    [[0.5, 3, 4, 8, 14], "results/full_1116_step128/benchmark_in_domain_scene_005_step128_mpc_mpc_500/videos/traj_000.mp4"],
    [[1, 4, 7.5, 13, 14], "results/full_1116_step128/benchmark_in_domain_scene_006_step128_mpc_mpc_500/videos/traj_000.mp4"]
]
for i, (seconds, video_path) in enumerate(data_list):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    for j, second in enumerate(seconds):
        frame_id = int(second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # 从第7帧开始（索引从0算）
        ret, frame = cap.read()
        cv2.imwrite(os.path.join(save_dir, f"{i}_{j}.png"), frame)
    cap.release()
