import numpy as np
import torch
import pdb


def rotation_error(q1, q2):
        """
        计算一批旋转矩阵之间的旋转误差（角度）。
        
        :param cur_R: 当前旋转矩阵 (batch_size x 3 x 3 PyTorch 张量)
        :param obs_R: 观测旋转矩阵 (batch_size x 3 x 3 PyTorch 张量)
        :return: 一批旋转矩阵之间的旋转误差（角度，以度为单位），返回大小为 (batch_size,)
        """
        q1 = q1 / torch.norm(q1, dim=1, keepdim=True)  # 归一化每个四元数
        q2 = q2 / torch.norm(q2, dim=1, keepdim=True)  # 归一化每个四元数
        
        # 计算每对四元数的点积
        dot_product = torch.sum(q1 * q2, dim=1)
        
        # 确保点积在 [-1, 1] 范围内，避免数值误差导致的 acos 计算问题
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # 计算旋转角度（弧度）
        angle_rad = 2 * torch.acos(torch.abs(dot_product))  # 计算每对四元数的夹角
        
        # 将角度从弧度转换为度
        angle_deg = angle_rad * (180.0 / torch.pi)
        
        return angle_deg
    
def translation_error(cur_T, obs_T):
    """
    计算一批平移向量之间的误差（欧氏距离）。
    
    :param cur_T: 当前平移向量 (batch_size x 3 PyTorch 张量)
    :param obs_T: 观测平移向量 (batch_size x 3 PyTorch 张量)
    :return: 一批平移向量之间的误差，返回大小为 (batch_size,)
    """
    return torch.linalg.norm(cur_T - obs_T, dim=1)
    
    
def cal_campose_error(pred_pose, target_pose):
    """
    计算总体损失，结合旋转和平移误差。
    
    :param cur_R: 当前旋转矩阵 (batch_size x 3 x 3 PyTorch 张量)
    :param obs_R: 观测旋转矩阵 (batch_size x 3 x 3 PyTorch 张量)
    :param cur_T: 当前平移向量 (batch_size x 3 PyTorch 张量)
    :param obs_T: 观测平移向量 (batch_size x 3 PyTorch 张量)
    :return: 总体损失值
    """
    rot_loss = rotation_error(pred_pose[:, 3:], target_pose[:, 3:])
    tran_loss = translation_error(pred_pose[:, :3], target_pose[:, :3])
    # total_loss = lambda_r * rot_loss.mean() + lambda_t * tran_loss.mean()
    return rot_loss, tran_loss

def calculate_errors_4x4(M1, M2):
    # 提取旋转矩阵和平移向量
    R1, T1 = M1[:3, :3], M1[:3, 3]
    R2, T2 = M2[:3, :3], M2[:3, 3]
    
    # 平移误差
    translation_error = np.linalg.norm(abs(T1) - abs(T2))
    
    # 旋转误差
    delta_R = R1.T @ R2
    rotation_error = np.arccos(np.clip((np.trace(delta_R) - 1) / 2, -1, 1))
    
    # 返回结果
    return translation_error, np.degrees(rotation_error)