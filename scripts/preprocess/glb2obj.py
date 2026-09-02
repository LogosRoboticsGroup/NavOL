import trimesh
import argparse
import os

def convert_glb_to_obj(glb_path, obj_path=None):
    """
    将GLB文件转换为OBJ文件
    
    参数:
    glb_path (str): GLB文件的路径
    obj_path (str, 可选): 输出OBJ文件的路径。如果未指定，将使用与GLB相同的名称和位置
    """
    try:
        # 加载GLB文件
        mesh = trimesh.load(glb_path)
        
        # 如果未指定输出路径，则生成默认路径
        if obj_path is None:
            # 获取文件名（不含扩展名）
            file_name = os.path.splitext(os.path.basename(glb_path))[0]
            # 获取目录路径
            dir_path = os.path.dirname(glb_path)
            # 组合成默认的OBJ路径
            obj_path = os.path.join(dir_path, f"{file_name}.obj")
        
        # 导出为OBJ文件
        mesh.export(obj_path, file_type='obj')
        
        print(f"成功将 {glb_path} 转换为 {obj_path}")
        return True
        
    except Exception as e:
        print(f"转换失败: {str(e)}")
        return False

def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='将GLB文件转换为OBJ文件')
    parser.add_argument('input', help='输入的GLB文件路径')
    parser.add_argument('-o', '--output', help='输出的OBJ文件路径（可选）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在")
        return
    
    # 检查输入文件是否为GLB格式
    if not args.input.lower().endswith('.glb'):
        print(f"错误: 输入文件 '{args.input}' 不是GLB格式")
        return
    
    # 执行转换
    convert_glb_to_obj(args.input, args.output)

if __name__ == "__main__":
    main()
