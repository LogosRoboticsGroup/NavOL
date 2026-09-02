import os

save_dir = "results_vis/figures_supp_videos"
os.makedirs(save_dir, exist_ok=True)
data_list = {
    "cluttered_easy": [
        "results/full_1116_step128/cluttered_easy_easy_0_step128_mpc_mpc_500/videos/traj_024.mp4",
        "results/full_1116_step128/cluttered_easy_easy_9_step128_mpc_mpc_500/videos/traj_075.mp4",
    ],
    "cluttered_hard": [
        "results/full_1116_step128/cluttered_hard_hard_0_step128_mpc_mpc_500/videos/traj_008.mp4",
        "results/full_1116_step128/cluttered_hard_hard_9_step128_mpc_mpc_500/videos/traj_001.mp4",
    ],
    "intern_commercial":[
        "results/full_1116_step128/internscenes_commercial_MV4AFHQKTKJZ2AABAAAAADQ8_usd_step128_mpc_mpc_500/videos/traj_008.mp4",
        "results/full_1116_step128/internscenes_commercial_MWF4WLIKTIFZIAABAAAAACY8_usd_step128_mpc_mpc_500/videos/traj_000.mp4",
    ],
    "intern_home": [
        "results/full_1116_step128/internscenes_home_MVUCSQAKTKJ5EAABAAAAABI8_usd_step128_mpc_mpc_500/videos/traj_022.mp4",
        "results/full_1116_step128/internscenes_home_MVUHLWYKTKJ5EAABAAAAAAI8_usd_step128_mpc_mpc_1000/videos/traj_007.mp4",
    ],
    "in_domain": [
        "results/full_1116_step128/benchmark_in_domain_scene_002_step128_mpc_mpc_500/videos/traj_057.mp4",
        "results/full_1116_step128/benchmark_in_domain_scene_003_step128_mpc_mpc_500/videos/traj_000.mp4",
    ],
    "out_domain": [
        "results/full_1116_step128/benchmark_out_domain_scene_000_step128_mpc_mpc_500/videos/traj_003.mp4",
        "results/full_1116_step128/benchmark_out_domain_scene_001_step128_mpc_mpc_500/videos/traj_026.mp4",
    ],
}

for scene_type in data_list:
    scene_data = data_list[scene_type]
    for i, video_path in enumerate(scene_data):
        cmd = f"cp {video_path} {os.path.join(save_dir, f'{scene_type}_{i}.mp4')}"
        os.system(cmd)