import os

import cv2
import sys

save_dir = "results_vis/figures_supp_cluttered"
os.makedirs(save_dir, exist_ok=True)
data_list = [
    [[0.2, 1, 7, 23, 29], "results/full_1116_step128/cluttered_easy_easy_0_step128_mpc_mpc_500/videos/traj_024.mp4"],
    [[0.6, 2, 10, 14, 20], "results/full_1116_step128/cluttered_easy_easy_9_step128_mpc_mpc_500/videos/traj_075.mp4"],
    [[4, 12, 16, 28, 32], "results/full_1116_step128/cluttered_hard_hard_0_step128_mpc_mpc_500/videos/traj_008.mp4"],
    [[2, 3.7, 5, 12, 15], "results/full_1116_step128/cluttered_hard_hard_9_step128_mpc_mpc_500/videos/traj_001.mp4"]
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
