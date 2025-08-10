import numpy as np
import pdb
from math import sin, cos, pi
from loader.loader_helper import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix, normalize_quaternion, random_pose_around_fixed_norm, relative_pose_vit, view_matrix, simulate_camera_motion

def source_per(output_filename):
    view = simulate_camera_motion()
    with open(output_filename, 'w') as file:
        for i in range(len(view)):
            def to_scientific(x):
                return f"{x:.9e}"

            # 将矩阵的每个元素转换为科学计数法格式
            vectorized_func = np.vectorize(to_scientific)
            matrix_scientific = vectorized_func(view[i])
            matrix_float64 = matrix_scientific.astype(np.float64)

            pose = np.array(matrix_float64).squeeze(0) # 每次取不同的矩阵

            output_filename = output_filename
            # for i in range(len(view)):
            # 保存rela_pose
            file.write(f"source_pose_{i}:\n")
            np.savetxt(file, pose, fmt='%.9e', delimiter=' ')
            file.write("\n")

def source_list(scene_name):
    output_filename = f'/home/whao/pose_eatimate/dataset/db/Ablation/24-source/Gaussian_Source/{scene_name}/source.txt'

    # source_per(output_filename)

    # 初始化一个空列表来存储4x4矩阵
    pose_list = []

    # 读取txt文件
    with open(output_filename, 'r') as file:
        lines = file.readlines()
        
        # 临时存储当前4x4矩阵数据
        current_pose = []
        
        for line in lines:
            line = line.strip()  # 去除空格和换行符
            
            # 如果这一行包含 'source_pose'，则表示是一个新的矩阵块
            if 'source_pose' in line:
                # 如果当前矩阵存在且非空，加入到列表
                if current_pose:
                    pose_list.append(np.array(current_pose))
                    current_pose = []  # 清空临时矩阵
            else:
                # 如果这一行不是 'source_pose'，将数据行拆分并加入矩阵
                if line:
                    # 将一行的数字字符串转换为浮动类型并加入当前矩阵
                    current_pose.append([float(x) for x in line.split()])
        
        # 最后一个矩阵块处理
        if current_pose:
            pose_list.append(np.array(current_pose))
    return pose_list