import torch
import pdb

class PoseLossCalculator:
    def __init__(self, lambda_r=1.0, lambda_t=30, lambda_s=10):
        """
        初始化计算器。
        
        :param lambda_r: 旋转误差的权重。
        :param lambda_t: 平移误差的权重。
        """
        self.lambda_r = lambda_r
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s
    

    def rotation_error(self, q1, q2):
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
    
    def translation_error(self, cur_T, obs_T):
        """
        计算一批平移向量之间的误差（欧氏距离）。
        
        :param cur_T: 当前平移向量 (batch_size x 3 PyTorch 张量)
        :param obs_T: 观测平移向量 (batch_size x 3 PyTorch 张量)
        :return: 一批平移向量之间的误差，返回大小为 (batch_size,)
        """
        return torch.linalg.norm(cur_T - obs_T, dim=1)
    
    def scale_error(self, cur_T, obs_T):
        """
        计算一批平移向量之间的误差（欧氏距离）。
        
        :param cur_T: 当前平移向量 (batch_size x 3 PyTorch 张量)
        :param obs_T: 观测平移向量 (batch_size x 3 PyTorch 张量)
        :return: 一批平移向量之间的误差，返回大小为 (batch_size,)
        """
        scale=torch.abs(torch.linalg.norm(cur_T, dim=1)-torch.linalg.norm(obs_T, dim=1))
        return scale
    
    
    def __call__(self, pred_pose, target_pose):
        """
        计算总体损失，结合旋转和平移误差。
        
        :param cur_R: 当前旋转矩阵 (batch_size x 3 x 3 PyTorch 张量)
        :param obs_R: 观测旋转矩阵 (batch_size x 3 x 3 PyTorch 张量)
        :param cur_T: 当前平移向量 (batch_size x 3 PyTorch 张量)
        :param obs_T: 观测平移向量 (batch_size x 3 PyTorch 张量)
        :return: 总体损失值
        """
        rot_loss = self.rotation_error(pred_pose[:, 3:], target_pose[:, 3:])
        tran_loss = self.translation_error(pred_pose[:, :3], target_pose[:, :3])
        scale_loss = self.scale_error(pred_pose[:, :3], target_pose[:, :3])
        total_loss = self.lambda_r * rot_loss.mean() + self.lambda_t * tran_loss.mean() + self.lambda_s * scale_loss.mean() 
        return total_loss
