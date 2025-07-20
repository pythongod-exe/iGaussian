import torch
import numpy as np

def relative_pose(start_pose_w2c_pred, start_pose_w2c_gt):
    start_pose_c2w_pred=np.linalg.inv(np.array(start_pose_w2c_pred))
    start_pose_c2w_gt=np.linalg.inv(np.array(start_pose_w2c_gt))
    """
    计算两个位姿的相对位姿。
    参数:
        pose1 (torch.Tensor): 第一个位姿 [x, y, z, qx, qy, qz, qw]，形状为 (7,)
        pose2 (torch.Tensor): 第二个位姿 [x, y, z, qx, qy, qz, qw]，形状为 (7,)
    返回:
        torch.Tensor: 相对位姿 [x, y, z, qx, qy, qz, qw]
    """
    pose1_inv = np.linalg.inv(start_pose_c2w_gt)
    relative_matrix = np.dot(pose1_inv, start_pose_c2w_pred)  

    return relative_matrix

def quaternion_to_pose_matrix(q, t):
    # 提取四元数和平移向量
    qx, qy, qz, qw = q
    x, y, z = t

    # 四元数转旋转矩阵
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])

    # 构建4x4位姿矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T