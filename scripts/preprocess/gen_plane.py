import trimesh

def generate_z_up_plane(filename, width=10, depth=10, divisions=10):
    """
    生成Z轴朝上的平面OBJ文件（即平面位于XY平面，Z轴垂直向上）
    
    参数:
    filename: 保存的文件名
    width: 平面X方向宽度
    depth: 平面Y方向深度
    divisions: 分割数量，决定平面的精细度
    """
    # 计算每个小格子的尺寸
    step_x = width / divisions
    step_y = depth / divisions
    
    vertices = []
    faces = []
    
    # 生成顶点 - Z值为0，使平面位于XY平面，Z轴朝上
    for y in range(divisions + 1):
        for x in range(divisions + 1):
            # 计算顶点坐标，z=0表示平面，Z轴垂直向上
            vertex_x = (x * step_x) - (width / 2)  # 居中平面
            vertex_y = (y * step_y) - (depth / 2)  # 居中平面
            vertex_z = 0.0  # Z值固定为0，形成平面
            vertices.append([vertex_x, vertex_y, vertex_z])
    
    # 生成三角形面片
    for y in range(divisions):
        for x in range(divisions):
            # 计算当前格子四个顶点的索引
            v0 = x + y * (divisions + 1)
            v1 = (x + 1) + y * (divisions + 1)
            v2 = (x + 1) + (y + 1) * (divisions + 1)
            v3 = x + (y + 1) * (divisions + 1)
            
            # 每个格子分成两个三角形
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    
    # 创建trimesh网格对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 保存为OBJ文件
    mesh.export(filename)
    
    print(f"已生成Z轴朝上的平面OBJ文件: {filename}")
    print(f"顶点数量: {len(vertices)}")
    print(f"三角形面片数量: {len(faces)}")

# 生成一个10x10的Z轴朝上平面，分成10x10的网格，保存为z_up_plane.obj
generate_z_up_plane("plane.obj", width=10, depth=10, divisions=10)
    