import bpy
import bmesh
import mathutils
import math
import sys
import os
import glob
import json
import argparse

def clear_scene():
    """Clear all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

def flip_top_bottom_faces(obj):
    """只翻转最高和最低平面的法向量"""
    # 确保在物体模式
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # 获取物体的边界框
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_y = min(v.y for v in bbox)
    max_y = max(v.y for v in bbox)
    y_range = max_y - min_y
    print("=="*20)
    print(y_range)
    print("=="*20)


    # 定义顶部和底部的阈值（边界框高度的10%范围内）
    top_threshold = max_y - y_range * 0.1
    bottom_threshold = min_y + y_range * 0.1

    print(f"Object {obj.name} - Y range: {min_y:.3f} to {max_y:.3f}")
    print(f"Top threshold: {top_threshold:.3f}, Bottom threshold: {bottom_threshold:.3f}")
    
    # 获取mesh数据
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    
    # 确保面的法向量已更新
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    
    # 找到顶部和底部的面
    faces_to_flip = []
    
    for face in bm.faces:
        # 计算面的中心点在世界坐标下的位置
        face_center = obj.matrix_world @ face.calc_center_median()
        face_y = face_center.y

        # 获取法向量
        normal = face.normal
        
        # 检查面是否在顶部或底部区域，并且是水平面
        is_at_top = face_y >= top_threshold
        is_at_bottom = face_y <= bottom_threshold

        if (normal.y > 0.8 and is_at_top) or (normal.y < -0.8 and is_at_bottom):
            faces_to_flip.append(face)

    # 翻转选中的面
    for face in faces_to_flip:
        face.normal_flip()
    
    # 更新mesh
    bm.to_mesh(obj.data)
    bm.free()
    
    # 更新对象
    obj.data.update()
    
    print(f"Flipped {len(faces_to_flip)} top/bottom faces for {obj.name}")
    print(top_threshold, bottom_threshold)

def scale_to_target_height(obj, target_height=5):
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_y = min(v.y for v in bbox)
    max_y = max(v.y for v in bbox)
    current_height = max_y - min_y
    scale_factor = target_height / current_height
    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return scale_factor

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

def align_min_z_to_zero(obj):
    """将对象世界包围盒的最低点对齐到 0"""
    # 计算世界坐标包围盒
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    max_y = max(v.y for v in bbox)  # y轴朝下
    # 直接在世界矩阵上平移，避免父子关系/原点影响
    m = obj.matrix_world.copy()
    m.translation.y -= max_y
    obj.matrix_world = m
    # 固化位移
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

def convert(input_path, output_path, output_path_navmesh, target_height):
    """Convert GLB from Y-up left-handed to Z-up right-handed while preserving textures"""
    
    print(f"Starting conversion: {input_path}")
    clear_scene()
    
    # 导入
    bpy.ops.import_scene.gltf(filepath=input_path)
    print("GLB imported successfully")
    
    # 合并
    merged = merge_all_meshes()
    print(f"Merged object: {merged.name}")
    
    bpy.ops.object.select_all(action='DESELECT')
    merged.select_set(True)
    bpy.context.view_layer.objects.active = merged
    rot_x = mathutils.Matrix.Rotation(math.radians(90.0), 4, 'X')
    merged.matrix_world = rot_x @ merged.matrix_world
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
    
    flip_top_bottom_faces(merged)
    
    scale_to_target_height(merged, target_height=target_height)
    
    align_min_z_to_zero(merged)
    
    bpy.ops.object.select_all(action='SELECT')
    # Export as GLB with texture preservation
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        use_selection=True,
        export_format='GLB',
        export_materials='EXPORT',
        export_texcoords=True,
        export_normals=True,
        export_tangents=False,
    )
    print(f"Converted model saved to: {output_path}")
    
    # Export as GLB with texture preservation
    bpy.ops.export_scene.gltf(
        filepath=output_path_navmesh,
        use_selection=True,
        export_format='GLB',
        export_materials='NONE',
        export_texcoords=False,
        export_normals=False,
        export_tangents=False,
    )
    print(f"Converted navmesh model saved to: {output_path_navmesh}")

    return True

def process_scenes(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    success = convert(args.input_path, args.output_path, args.output_navmesh_path, args.target_height)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GLB files from Y-up left-handed to Z-up right-handed.")
    parser.add_argument("--input_path", type=str, default="", help="Input GLB file path.")
    parser.add_argument("--output_path", type=str, default="", help="Output GLB file path.")
    parser.add_argument("--output_navmesh_path", type=str, default="", help="Output GLB file path.")
    parser.add_argument("--target_height", type=float, default=5.0, help="Target height to scale the models to.")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    args = parser.parse_args(argv)
    
    process_scenes(args)