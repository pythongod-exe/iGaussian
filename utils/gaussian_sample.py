import torch
from torchvision.utils import save_image
from scipy.spatial.transform import Rotation as R
from rendering.Guassion_render_train import rendering

import cv2
import numpy as np

import pdb

angle_x=2

def generate_poses(pred_Q, pred_T):
    pred_Q=pred_Q.squeeze(0)
    pred_T=pred_T.squeeze(0)
    translate_vectors=generate_cone_vectors(pred_T, angle_x, num_vectors=3)
    translate_vectors=torch.cat((translate_vectors, pred_T.unsqueeze(0)),dim=0)
    rotation_vector=pred_Q.unsqueeze(0).expand(translate_vectors.size(0), -1)

    return translate_vectors, rotation_vector

def generate_cone_vectors(V0, angle_x, num_vectors=5):
    # 确保输入向量是单位向量
    V_unnorm = V0
    V0 = V0 / V0.norm()
    
    # 将角度转换为弧度
    angle_x_rad = torch.tensor(angle_x * (torch.pi / 180.0))
    
    # 计算夹角的余弦和正弦
    cos_angle = torch.cos(angle_x_rad)
    sin_angle = torch.sin(angle_x_rad)

    # 生成新的向量
    vectors = []
    for i in range(num_vectors):
        # 计算底面圆上的角度
        theta = torch.tensor(2 * torch.pi * i / num_vectors).to(V0.device)
        
        # 计算新向量的坐标
        new_vector = torch.tensor([
            cos_angle * V0[0] + sin_angle * torch.cos(theta),
            cos_angle * V0[1] + sin_angle * torch.sin(theta),
            cos_angle * V0[2]
        ]).to(V0.device)
        
        # 归一化为单位向量
        vectors.append((new_vector / new_vector.norm())*V_unnorm.norm())

    return torch.stack(vectors)

def pose_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    """Convert pose to 4x4 transformation matrix using PyTorch."""
    rotation_matrix = torch.tensor(R.from_quat([qx, qy, qz, qw]).as_matrix(), dtype=torch.float32)
    transformation_matrix = torch.eye(4, dtype=torch.float32)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)
    return transformation_matrix

def matrix_to_pose(matrix):
    """Convert 4x4 transformation matrix back to pose format [tx, ty, tz, qx, qy, qz, qw] using PyTorch."""
    tx, ty, tz = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3].numpy()  # Convert to NumPy for quaternion conversion
    rotation = R.from_matrix(rotation_matrix)
    qx, qy, qz, qw = rotation.as_quat()
    return [tx.item(), ty.item(), tz.item(), qx, qy, qz, qw]



def save_depth_visualization(depth_map, output_path, colormap=cv2.COLORMAP_JET):
    # 归一化深度图，使其深度值在0到255之间
    normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 应用伪彩色映射，将深度值转换为颜色
    color_mapped_depth = cv2.applyColorMap(normalized_depth, colormap)
    
    # 保存结果
    cv2.imwrite(output_path, color_mapped_depth)


