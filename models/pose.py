import torch
import torch.nn as nn
import torchvision.models as models

from .pose_forward import SpatialTransformerBlock
from .encoder_decoder import Decoderx8, Encoderx8Simp
from .pose_forward import CrossAttention
from .poes2vector import PoseToFeature
from .pose_decoder import FeatureMapDecoder
import pdb


class Pose(nn.Module):
    def __init__(self, channel=256, height=384, width=512, n_layers=8, tmp_size=20):
        super(Pose, self).__init__()
        self.transformers = nn.ModuleList(
            [SpatialTransformerBlock(channel) for _ in range(n_layers)])
        self.encoder = Encoderx8Simp(6, channel, use_norm=False, use_bias=False)
        self.decoder = Decoderx8(channel, 6)
        # self.downsample = nn.Upsample(size=(200, 200), mode='bilinear', align_corners=True)

    def forward(self, concate_feature): # [7, 6, 800, 800]
        # concate_feature = self.downsample(concate_feature)
        concate_feature = self.encoder(concate_feature) # [7, 128, 100, 100]

        for module in self.transformers:
            feature_map = module(concate_feature) # [7, 128, 100, 100]

        output = self.decoder(feature_map) # [7, 6, 200, 200]

        return output
    
class Pose_ATTN(nn.Module):
    def __init__(self, channel=256, height=400, width=400, n_layers=3, tmp_size=20):
        super(Pose_ATTN, self).__init__()
        self.transformers_attn = nn.ModuleList(
            [CrossAttention(channel) for _ in range(n_layers)])

        self.encoderk = Encoderx8Simp(6, channel, use_norm=True, use_bias=False)
        self.encoderv = Encoderx8Simp(6, channel, use_norm=False, use_bias=False)
        self.decoder = Decoderx8(channel, 3) # channel 128

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(128, 128 * 25 * 25)

    def forward(self, k_frames, q_frames, v_frames): # [4, 6, 200, 400] [4, 128] [4, 6, 200, 400]
        k = self.encoderk(k_frames) # [4, 128, 50, 50]
        v = self.encoderv(v_frames) # [4, 128, 50, 50]
        b, c, h, w = k.shape

        q1 = self.fc(q_frames)
        q1 = q1.view(q1.shape[0], 128, 25, 25)

        for module in self.transformers_attn:
            v = module(k, q1, v) # [1, 128, 2500, 1] [1, 128, 2500, 1] [1, 128, 48, 64] [1, 128, 2500, 1]

        output = v.squeeze(-1)
        return output
    
class WeightPredictorModule(nn.Module):
    def __init__(self, feature_dim_1, feature_dim_2):
        super(WeightPredictorModule, self).__init__()
        # 定义用于预测 Q 和 T 的两个独立权重预测器
        self.weight_predictor = nn.Sequential(
            nn.Linear(feature_dim_1, feature_dim_2),
            nn.ReLU(),
            nn.Linear(feature_dim_2, feature_dim_2),  # 输出 Q 权重
            nn.ReLU(), 
            nn.Linear(feature_dim_2, 1),
        )

    def forward(self, feature_map, scale_factor=10): # [4, 3, 400, 400]
        flattened_feature_map = feature_map.view(feature_map.size(0), -1)  # [batch_size, feature_dim] [4, 480000]

        weight = self.weight_predictor(flattened_feature_map) # [batch_size, 1]

        weight_normalized = torch.softmax(weight, dim=0) # [batch_size, 1]

        return weight_normalized
    
class Pose_Pred(nn.Module):
    def __init__(self, hparams):
        super(Pose_Pred, self).__init__()

        # backbone:resnet18
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(self.resnet18.children())[:-2])
        self.conv_out = nn.Conv2d(in_channels=512, out_channels=6, kernel_size=1, stride=1, padding=0, bias=False)
        self.upsample = nn.Upsample(size=(200, 200), mode='bilinear', align_corners=True)

        # backbone:adela
        # self.model = Pose(height=hparams.input_height, width=hparams.input_width, tmp_size=hparams.frnt_rng,
        #                                 channel=hparams.n_channel, n_layers=hparams.n_layers)
        
        # attention
        self.model_attention = Pose_ATTN(height=hparams.input_height, width=hparams.input_width, tmp_size=hparams.frnt_rng,
                                                        channel=hparams.n_channel, n_layers=hparams.n_layers)
        
        self.map2vector = PoseToFeature()
        self.vector2map = FeatureMapDecoder()
        self.weight_module = WeightPredictorModule(feature_dim_1=200 * 200 * 6, feature_dim_2=200 * 6)
        
    def forward(self, target_imgs, source_imgs, source_poses):
        target_imgs=target_imgs.squeeze(0)
        source_imgs=source_imgs.squeeze(0)
        source_poses = source_poses.squeeze(0)

        concate_feature = torch.cat((target_imgs, source_imgs), dim=1) # [N, 6, 400, 400]

        x = self.backbone(concate_feature) # [N, 512, 13, 13]
        x = self.conv_out(x) # [N, 6, 13, 13]
        feature_map = self.upsample(x) # [N, 6, 200, 200]

        # concate_feature = self.upsample(concate_feature)
        # feature_map = self.model(concate_feature) # [7, 6, 200, 200]

        weight = self.weight_module(feature_map) # [N, 1] [N, 1]

        feature_vector = self.map2vector(source_poses) # [N, 128]
        feature_attn = self.model_attention(feature_map, feature_vector, feature_map) # [N, 128, 25, 25]
        pred_poses = self.vector2map(feature_attn)

        pred_pose = torch.sum(pred_poses * weight, dim=0).unsqueeze(0)

        return pred_pose
