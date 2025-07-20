import torch
import torch.nn as nn
import pdb

# 定义单层的 MLP 类
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),  # 输入维度 -> 输出维度
            nn.LeakyReLU()  # 激活函数
        )
    
    def forward(self, x):
        return self.fc(x)
    
class FeatureMapDecoder(nn.Module):
    def __init__(self):
        super(FeatureMapDecoder, self).__init__()

        input_shape = 128 * 25 * 25  # 展平后的输入维度
        intermediate_dims = [1024, 256]  # 中间层的维度
        output_dim = 3  # 最终输出维度
        
        # 初始化三个 MLP
        self.mlp1 = SimpleMLP(input_dim=input_shape, output_dim=intermediate_dims[0])
        self.mlp2 = SimpleMLP(input_dim=intermediate_dims[0], output_dim=intermediate_dims[1])
        self.mlp3 = SimpleMLP(input_dim=intermediate_dims[1], output_dim=7)
        
        self.flatten = nn.Flatten()
    
    def forward(self, x): # [1, 128, 2500]
        b, _, _, _ = x.shape
        x = self.flatten(x)  # 将输入展平为一维向量
        
        # 依次通过三个 MLP
        feature1 = self.mlp1(x)
        feature2 = self.mlp2(feature1)
        feature3 = self.mlp3(feature2)
        # final_feature = feature3_t.view(b, 3)  # 将最终特征向量重塑为 4x4 的姿态矩阵
        
        return feature3
    
