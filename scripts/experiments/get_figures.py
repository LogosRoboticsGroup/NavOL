import os

import cv2
import sys

save_dir = "results_vis/figures"
os.makedirs(save_dir, exist_ok=True)
video_path = "results_vis/001/videos/traj_001.mp4"
video_depth_path = "results_vis/001/videos/traj_001_depth.mp4"
cap = cv2.VideoCapture(video_path)
cap_depth = cv2.VideoCapture(video_depth_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
seconds = [0.6,3,20,30,40]
for i, second in enumerate(seconds):
    frame_id = int(second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # 从第7帧开始（索引从0算）
    ret, frame = cap.read()
    cap_depth.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # 从第7帧开始（索引从0算）
    _, frame_depth = cap_depth.read()
    cv2.imwrite(os.path.join(save_dir, f"01_{i}.png"), frame)
    cv2.imwrite(os.path.join(save_dir, f"01_{i}_depth.png"), frame_depth)
cap.release()
cap_depth.release()


video_path = "results_vis/002/benchmark_out_domain_scene_000/videos/traj_000.mp4"
video_depth_path = "results_vis/002/benchmark_out_domain_scene_000/videos/traj_000_depth.mp4"
cap = cv2.VideoCapture(video_path)
cap_depth = cv2.VideoCapture(video_depth_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
seconds = [0.5,14,17,22,38]
for i, second in enumerate(seconds):
    frame_id = int(second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # 从第7帧开始（索引从0算）
    ret, frame = cap.read()
    cap_depth.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # 从第7帧开始（索引从0算）
    _, frame_depth = cap_depth.read()
    cv2.imwrite(os.path.join(save_dir, f"02_{i}.png"), frame)
    cv2.imwrite(os.path.join(save_dir, f"02_{i}_depth.png"), frame_depth)
cap.release()
cap_depth.release()



video_path = "results_vis/003/cluttered_easy_easy_0/videos/traj_001.mp4"
video_depth_path = "results_vis/003/cluttered_easy_easy_0/videos/traj_001_depth.mp4"
cap = cv2.VideoCapture(video_path)
cap_depth = cv2.VideoCapture(video_depth_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
seconds = [0,2,18,36,51]
for i, second in enumerate(seconds):
    frame_id = int(second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # 从第7帧开始（索引从0算）
    ret, frame = cap.read()
    cap_depth.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # 从第7帧开始（索引从0算）
    _, frame_depth = cap_depth.read()
    cv2.imwrite(os.path.join(save_dir, f"03_{i}.png"), frame)
    cv2.imwrite(os.path.join(save_dir, f"03_{i}_depth.png"), frame_depth)
cap.release()
cap_depth.release()