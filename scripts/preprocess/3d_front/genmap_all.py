import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--scene_dir", type=str, default="source/scene_data_3d_front/3d_front_scene_1000")
args_cli = parser.parse_args()

save_path = os.path.join(args_cli.scene_dir, "maps")
mesh_dir = os.path.join(args_cli.scene_dir, "navmesh_scenes")
scenes = sorted([os.path.join(mesh_dir, p) for p in os.listdir(mesh_dir) if p.endswith(".glb")])
# scenes = scenes[30:100]
print(f"Total scenes to process: {len(scenes)}")
for scene_path in tqdm(scenes):
    if os.path.exists(os.path.join(save_path, os.path.basename(scene_path)[:-4]+'.png')):
        continue
    cmd = f"python scripts/preprocess/3d_front/genmap.py --scene_path {scene_path} --save_path {save_path}"
    os.system(cmd)