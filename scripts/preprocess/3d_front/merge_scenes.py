import bpy
import bmesh
import mathutils
import math
import sys
import os
import json
import numpy as np
from pathlib import Path
import argparse

def clear_scene():
    """Clear all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
            
def merge_all_meshes():
    bpy.ops.object.mode_set(mode='OBJECT')
    # 选择所有 MESH
    for obj in bpy.context.scene.objects:
        obj.select_set(obj.type == 'MESH')
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        return None
    # 设定活动对象（作为 join 目标）
    bpy.context.view_layer.objects.active = meshes[0]
    # 合并
    bpy.ops.object.join()
    merged = bpy.context.view_layer.objects.active
    merged.name = "MergedMesh"
    return merged

def merge_scenes(args):
    """合并well文件夹下的所有_z_up.glb文件"""
    with open(os.path.join(args.scene_path, "index.json"), 'r') as f:
        scene_dict = json.load(f)
    
    scene_dir = os.path.join(args.scene_path, "scenes")
    
    if os.path.exists(os.path.join(args.scene_path, "selected.json")):
        with open(os.path.join(args.scene_path, "selected.json"), 'r') as f:
            selected_indices = json.load(f)
        selected_scene_names = [f"scene_{i:03d}" for i in selected_indices]
        scene_dict = {name: scene_dict[name] for name in selected_scene_names}
        print(f"Selected {len(selected_scene_names)} scenes for merging.")
    else:
        for scene_name in scene_dict.keys():
            scene_path = os.path.join(scene_dir, scene_name + ".glb")
            assert os.path.exists(scene_path), f"Scene file not found: {scene_path}"
    
    clear_scene()
    
    transform_records = {}
    spacing = 10.0
    
    for i, scene_name in enumerate(scene_dict.keys()):
        print(f"\nProcessing {i+1}/{len(scene_dict)}: {scene_name}")
        scene_path = os.path.join(scene_dir, scene_name + ".glb")
        
        bpy.ops.import_scene.gltf(filepath=scene_path)
        imported_scenes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        assert len(imported_scenes) == 1, f"Expected one mesh object per scene, found {len(imported_scenes)} in {scene_name}"
        
        obj = imported_scenes[0]
        m = obj.matrix_world.copy()
        m.translation.y -= spacing * i
        obj.matrix_world = m
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
        
    merge_all_meshes()
    bpy.ops.object.select_all(action='SELECT')
    
    output_path = os.path.join(args.scene_path, "scene.glb")
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        use_selection=True,
        export_format='GLB',
        export_materials='EXPORT',
        export_texcoords=True,
        export_normals=True
    )
    print(f"\nMerged GLB saved to: {output_path}")
    print(f"Total scenes merged: {len(transform_records)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all _z_up.glb scenes in a well folder into a single GLB.")
    parser.add_argument("--data_root", type=str, default="source/scene_data_3d_front", help="Path to the root directory containing scene folders.")
    parser.add_argument("--scene_path", type=str, default="source/scene_data_3d_front/3d_front_scene_1", help="Path to the scene index file.")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    args = parser.parse_args(argv)
    merge_scenes(args)
    