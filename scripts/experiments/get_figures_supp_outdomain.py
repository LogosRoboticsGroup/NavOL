import os

import cv2
import sys

save_dir = "results_vis/figures_supp_outdomain"
os.makedirs(save_dir, exist_ok=True)
data_list = [
    [[0.3, 4, 12, 14, 18.5], "results/full_1116_step128/benchmark_out_domain_scene_000_step128_mpc_mpc_500/videos/traj_003.mp4"],
    [[0.3, 3.8, 15, 18, 27], "results/full_1116_step128/benchmark_out_domain_scene_001_step128_mpc_mpc_500/videos/traj_026.mp4"],
    [[1, 10, 12, 13, 15], "results/full_1116_step128/benchmark_out_domain_scene_002_step128_mpc_mpc_500/videos/traj_002.mp4"],
    [[1, 7, 11, 14, 15], "results/full_1116_step128/benchmark_out_domain_scene_004_step128_mpc_mpc_500/videos/traj_020.mp4"]
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
