import torch
import torch.nn as nn
import pdb

# 定义一个单层的 MLP 模块
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=16, output_dim=64):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),  # 输入维度 16 -> 输出维度 64
            nn.LeakyReLU()  # 激活函数
        )
    
    def forward(self, x):
        return self.fc(x)
    
class PoseToFeature(nn.Module):
    def __init__(self, input_dim=9, output_dim=64, final_output_dim=128):
        super(PoseToFeature, self).__init__()
        # 初始化三个 MLP
        self.mlp1 = SimpleMLP(input_dim=7, output_dim=32)
        self.mlp2 = SimpleMLP(input_dim=32, output_dim=64)
        self.mlp3 = SimpleMLP(input_dim=64, output_dim=128)
    
    def forward(self, x):
        x = x.view(x.size(0), -1) # 将输入展平为一维向量
        
        # 依次通过三个 MLP
        feature1 = self.mlp1(x)
        feature2 = self.mlp2(feature1)
        feature3 = self.mlp3(feature2)
        
        return feature3

