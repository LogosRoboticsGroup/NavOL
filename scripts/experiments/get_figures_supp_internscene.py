import os

import cv2
import sys

save_dir = "results_vis/figures_supp_internscene"
os.makedirs(save_dir, exist_ok=True)
data_list = [
    [[2, 13, 14, 17, 19], "results/full_1116_step128/internscenes_commercial_MV4AFHQKTKJZ2AABAAAAADQ8_usd_step128_mpc_mpc_500/videos/traj_008.mp4"],
    [[1.6, 2.3, 6, 11, 14], "results/full_1116_step128/internscenes_commercial_MWF4WLIKTIFZIAABAAAAADA8_usd_step128_mpc_mpc_500/videos/traj_072.mp4"],
    [[1, 4, 6, 8, 16], "results/full_1116_step128/internscenes_home_MVUCSQAKTKJ5EAABAAAAABI8_usd_step128_mpc_mpc_500/videos/traj_022.mp4"],
    [[1, 1.5, 3, 4, 5], "results/full_1116_step128/internscenes_home_MVUHLWYKTKJ5EAABAAAAAAI8_usd_step128_mpc_mpc_1000/videos/traj_007.mp4"]
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
