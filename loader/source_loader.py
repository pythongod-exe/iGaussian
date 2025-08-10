import sys
sys.path.append('/home/whao/pose_eatimate/dataset/db/Ablation/24-source')
import numpy as np
import pdb
from math import sin, cos, pi
from loader.loader_helper import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix, normalize_quaternion, random_pose_around_fixed_norm, relative_pose_vit, view_matrix, simulate_camera_motion

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from loader.blender_loader import BlenderDataset, BlenderDataset_val, BlenderDataset_test
# from loader.blender_loader import BlenderDataset_val
from models.pose import Pose_Pred
# from modules_vit.model import ViTEss
import argparse
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from argparse import ArgumentParser
from gaussian.arguments import ModelParams, PipelineParams,iComMaParams, get_combined_args
from gaussian.gaussian_renderer import GaussianModel
import torch
from rendering.train_sence import Scene_playroom, Scene_drjohnson
import pdb

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

    source_per(output_filename)

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

base_dirs_train = [
                    #    '/data1/whao_dataset/tandt_db/tandt_db/db/playroom', # 1267, 832
                       '/data1/whao_dataset/tandt_db/tandt_db/db/drjohnson',
                        ],

parser_playroom = ArgumentParser(description="Camera pose estimation parameters")
parser_playroom.add_argument("--quiet", action="store_true")
parser_playroom.add_argument("--output_path", default='output', type=str,help="output path")
parser_playroom.add_argument("--obs_img_index", default=0, type=int)
parser_playroom.add_argument("--delta", default="[30,10,5,0.1,0.2,0.3]", type=str)
parser_playroom.add_argument("--iteration", default=-1, type=int)
parser_playroom.add_argument("--model_dir", type=str, default="/data1/whao_model/3dgs_model/playroom", help="Path to the model directory")
parser_playroom.add_argument('--no_pos_encoding', action='store_true')
parser_playroom.add_argument('--noess', action='store_true')
parser_playroom.add_argument('--cross_features', action='store_true')
parser_playroom.add_argument('--use_single_softmax', action='store_true')  
parser_playroom.add_argument('--l1_pos_encoding', action='store_true')
parser_playroom.add_argument('--fusion_transformer', action="store_true", default=True)
parser_playroom.add_argument('--fc_hidden_size', type=int, default=512)
parser_playroom.add_argument('--pool_size', type=int, default=60)
parser_playroom.add_argument('--transformer_depth', type=int, default=6)

model_playroom = ModelParams(parser_playroom, sentinel=True)
pipeline_playroom = PipelineParams(parser_playroom)
icommaparams_playroom = iComMaParams(parser_playroom)
args_playroom = get_combined_args(parser_playroom)

args_playroom.data_device = torch.device('cuda:0') 

# # load gaussians
dataset_playroom = model_playroom.extract(args_playroom)
gaussians_playroom = GaussianModel(dataset_playroom.sh_degree)
Scene_playroom(gaussians_playroom)


parser_drjohnson = ArgumentParser(description="Camera pose estimation parameters")
parser_drjohnson.add_argument("--quiet", action="store_true")
parser_drjohnson.add_argument("--output_path", default='output', type=str,help="output path")
parser_drjohnson.add_argument("--obs_img_index", default=0, type=int)
parser_drjohnson.add_argument("--delta", default="[30,10,5,0.1,0.2,0.3]", type=str)
parser_drjohnson.add_argument("--iteration", default=-1, type=int)
parser_drjohnson.add_argument("--model_dir", type=str, default="/data1/whao_model/3dgs_model/drjohnson", help="Path to the model directory")
parser_drjohnson.add_argument('--no_pos_encoding', action='store_true')
parser_drjohnson.add_argument('--noess', action='store_true')
parser_drjohnson.add_argument('--cross_features', action='store_true')
parser_drjohnson.add_argument('--use_single_softmax', action='store_true')  
parser_drjohnson.add_argument('--l1_pos_encoding', action='store_true')
parser_drjohnson.add_argument('--fusion_transformer', action="store_true", default=True)
parser_drjohnson.add_argument('--fc_hidden_size', type=int, default=512)
parser_drjohnson.add_argument('--pool_size', type=int, default=60)
parser_drjohnson.add_argument('--transformer_depth', type=int, default=6)

model_drjohnson = ModelParams(parser_drjohnson, sentinel=True)
pipeline_drjohnson = PipelineParams(parser_drjohnson)
icommaparams_drjohnson = iComMaParams(parser_drjohnson)
args_drjohnson = get_combined_args(parser_drjohnson)

args_drjohnson.data_device = torch.device('cuda:0') 

# # load gaussians
dataset_drjohnson = model_drjohnson.extract(args_drjohnson)
gaussians_drjohnson = GaussianModel(dataset_drjohnson.sh_degree)
Scene_drjohnson(gaussians_drjohnson)


if __name__ == '__main__':
    import os
    import imageio 
    from rendering.Guassion_render_train import rendering, rendering_iter, rendering_loader
    for base_dir in base_dirs_train[0]:
        scene_name = os.path.basename(base_dir)  
        pose_list = source_list(scene_name)

        source_pose = []
        for i in range(len(pose_list)):
            pose = pose_list[i]
            R = pose[:3, :3]
            T = pose[:3,  3]
            quaternion = rotation_matrix_to_quaternion(R) # wxyz
            quaternion = normalize_quaternion(quaternion)
            poses = np.concatenate((T, quaternion), axis=0)
            poses = torch.tensor([poses])
            source_pose.append(poses)
        source_poses=torch.cat((source_pose),dim=0)

        if scene_name=='playroom':
            gaussians=gaussians_playroom
            camera_parameters = {
                "FovX": 1.0921066048300383,  # 水平视场角 (弧度)
                "FovY": 0.7608849519220179,  # 垂直视场角 (弧度)
                "orig_w": 1264,  # 原始图像宽度
                "orig_h": 832   # 原始图像高度
            }
        elif scene_name=='drjohnson':
            gaussians=gaussians_drjohnson
            camera_parameters = {
                "FovX": 1.1431171654164658,  # 水平视场角 (弧度)
                "FovY": 0.8006899358019289,  # 垂直视场角 (弧度)
                "orig_w": 1264,  # 原始图像宽度
                "orig_h": 832   # 原始图像高度
            }

        gaussian_images = rendering_loader(gaussians, source_poses[10:], camera_parameters)

        for i in range(len(gaussian_images)):
            # target_pose=torch.tensor([[ 7.5220e-03, -2.8307e+00,  2.8700e+00,  0,  -1.821e-03,   6.8539e-01,  -6.949e-01]]).float()
            # gaussian_images,target_pose_w2c = rendering_loader(gaussians, view_quat[[i]])
            target_imgs = torch.clamp(gaussian_images[i], min=0.0, max=1.0).cpu()
            target_img_filename = f'/home/whao/pose_eatimate/dataset/db/Ablation/24-source/Gaussian_Source/{scene_name}/source_{i+10}.png'
            rgb = target_imgs.clone().permute(1, 2, 0).cpu().detach().numpy()
            # 将渲染图像转换为8位像素格式
            to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
            rgb8 = to8b(rgb)
            imageio.imwrite(target_img_filename, rgb8)