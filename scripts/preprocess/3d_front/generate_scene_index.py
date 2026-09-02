import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--total_scenes", type=int, default=1, help="Total number of scenes to include in the index.")
parser.add_argument("--data_root", type=str, default="source/scene_data_3d_front/", help="Root directory for scene data.")
parser.add_argument("--benchmark", action='store_true', help="Whether the data root is for benchmark scenes.")
args = parser.parse_args()

root = args.data_root
total_scenes = args.total_scenes
raw_scene_path = os.path.join(root, "raw_scenes")
scenes = sorted([os.path.join(raw_scene_path, f) for f in os.listdir(raw_scene_path) if f.endswith(".glb")])
scenes = scenes[:total_scenes]

if args.benchmark:
    name = f"3d_front_scene"
else:
    name = f"3d_front_scene_{total_scenes}"
os.makedirs(os.path.join(root, name), exist_ok=True)
with open(os.path.join(root, name, "index.json"), "w") as f:
    output = {f"scene_{i:03d}": {"idx": i, "name": f"scene_{i:03d}.glb", "path": scenes[i]} for i in range(len(scenes))}
    json.dump(output, f, indent=4)