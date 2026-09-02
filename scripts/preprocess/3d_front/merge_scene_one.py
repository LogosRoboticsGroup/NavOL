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
    clear_scene()
    
    transform_records = {}
    bpy.ops.import_scene.gltf(filepath=args.input_path)
    imported_scenes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    obj = imported_scenes[0]
    m = obj.matrix_world.copy()
    obj.matrix_world = m
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
        
    merge_all_meshes()
    bpy.ops.object.select_all(action='SELECT')
    
    bpy.ops.export_scene.gltf(
        filepath=args.output_path,
        use_selection=True,
        export_format='GLB',
        export_materials='EXPORT',
        export_texcoords=True,
        export_normals=False
    )
    bpy.ops.export_scene.gltf(
        filepath=args.output_navmesh_path,
        use_selection=True,
        export_format='GLB',
        export_materials='NONE',
        export_texcoords=False,
        export_normals=False,
        export_tangents=False,
    )
    print(f"GLB saved to: {args.output_path}")
    print(f"Navmesh GLB saved to: {args.output_navmesh_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all _z_up.glb scenes in a well folder into a single GLB.")
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--output_navmesh_path", type=str, default=None)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    args = parser.parse_args(argv)
    merge_scenes(args)
    