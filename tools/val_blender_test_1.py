import sys
sys.path.append('/home/whao/pose_eatimate/Feed-forward_iGuassion-weight')
import os
import numpy as np
import copy
import time
import torch
import json
from torch.utils.data import Dataset,DataLoader
from models.util.load_model import load_model
from utils.config_util import load_config_module
from utils.calculate_error_utils_test import cal_campose_error
from gaussian.scene.cameras import Camera_Pose
import copy
import time
from tqdm import tqdm
import ast
import pdb
import cv2
import imageio
from argparse import ArgumentParser
from gaussian.arguments import ModelParams, PipelineParams,iComMaParams, get_combined_args
from gaussian.utils.general_utils import safe_state
from os import makedirs
from gaussian.gaussian_renderer import GaussianModel
# from gaussian.scene import Scene
from rendering.train_sence import Scene
from gaussian.utils.icomma_helper import load_LoFTR, get_pose_estimation_input, get_pose_estimation_input_copy, get_pose_estimation_input_1
from gaussian.run_source_imgs import camera_pose_estimation
from gaussian.run_source_imgs import half_pixle_sampling
from gaussian.run import readCamerasFromTransforms
from PIL import Image
from torchvision import transforms
from gaussian.utils.image_utils import to8b
# from rendering.rendering_estimate_pose import rendering
from rendering.Guassion_render_train import rendering
from scipy.spatial.transform import Rotation as R

import torchvision.transforms as T
from PIL import Image
import torch
from torchvision import transforms

# model_path = '/data1/whao_model/feed_forward_gaussian/best_model.pth' #model path of verification
model_path = '/data1/whao_model/pose_estimate/weight_4_true/best_model.pth' #model path of verification

''' parameter from config '''
config_file = './config_blender.py'
configs = load_config_module(config_file)

def combine_3dgs_rotation_translation(R_c2w, T_w2c):
    RT_w2c = np.eye(4)
    RT_w2c[:3, :3] = R_c2w.T
    RT_w2c[:3, 3] = T_w2c
    RT_c2w=np.linalg.inv(RT_w2c)
    return RT_c2w

def pose_transform(pose):
    c2w = pose
    c2w[:3, 1:3] *= -1
    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    gt_pose_c2w=combine_3dgs_rotation_translation(R,T)
    # start_pose_c2w =  trans_t_xyz(delta[3],delta[4],delta[5]) @ rot_phi(delta[0]/180.*np.pi) @ rot_theta(delta[1]/180.*np.pi) @ rot_psi(delta[2]/180.*np.pi)  @ gt_pose_c2w

    start_pose_w2c=torch.from_numpy(np.linalg.inv(gt_pose_c2w)).float()
    # pose_w2c = torch.matmul(deltaT, start_pose_w2c.inverse()).inverse()

    return start_pose_w2c.inverse().clone().cpu().detach().numpy()


def pose_transform_torch(pose):
    c2w = pose.clone()
    c2w[:3, 1:3] *= -1
    # get the world-to-camera transform and set R, T
    w2c = torch.linalg.inv(c2w)
    R = w2c[:3, :3].T  # 直接使用 .T 进行转置操作（等效于 np.transpose）
    
    # 提取平移向量 T
    T = w2c[:3, 3]

    # 组合旋转和平移成新的相机到世界的变换矩阵
    gt_pose_c2w = combine_3dgs_rotation_translation(R, T)  # 假设 combine_3dgs_rotation_translation 已被改写为 PyTorch 实现
    
    # 计算从相机到世界的起始变换并转换为 PyTorch 张量
    start_pose_w2c = torch.linalg.inv(gt_pose_c2w)  # 计算世界到相机的变换
    
    # 返回从相机到世界的变换，作为 NumPy 数组输出
    return start_pose_w2c  # 将张量转回 NumPy 数组
        

def val(target_imgs, source_imgs, source_Qs, source_Ts, target_Qs, target_Ts):
    model = configs.model()
    model = load_model(model, model_path)
    print(model_path)
    model.cuda()

    for i in range(target_imgs.shape[0]):
        target_img = target_imgs[i].unsqueeze(0).float().cuda()
        source_img = source_imgs[i].unsqueeze(0).float().cuda()
        target_Q = target_Qs[i].unsqueeze(0).float().cuda()
        target_T = target_Ts[i].unsqueeze(0).float().cuda()
        source_Q = source_Qs[i].unsqueeze(0).float().cuda()
        source_T = source_Ts[i].unsqueeze(0).float().cuda()
        # source_pose_ = source_pose[i].unsqueeze(0).float().cuda()

        start_time = time.time()

        # -------------------------------------------------------------------------------------------------- #
        target_img_path = '/home/whao/pose_eatimate/Feed-forward_iGuassion-weight/test_outputs/test_source_img.png'
        target_image = Image.open(target_img_path)
        transform = transforms.ToTensor()
        target_image_tensor = transform(target_image).unsqueeze(0).unsqueeze(0).cuda()

        source_img_path = '/home/whao/pose_eatimate/Feed-forward_iGuassion-weight/test_outputs/test_estimate_img.png'
        source_image = Image.open(source_img_path)
        transform = transforms.ToTensor()
        source_image_tensor = transform(source_image).unsqueeze(0).unsqueeze(0).cuda()

        source_Q = torch.tensor([[0.00734939, 0.93514982, -0.35411806, 0.00641837]]).unsqueeze(0).float().cuda()
        source_T = torch.tensor([[-0.08628344, 2.707209, 3.0042155]]).unsqueeze(0).float().cuda()
        pdb.set_trace()
        # -------------------------------------------------------------------------------------------------- #

        pred_Q, pred_T = model(target_image_tensor, source_image_tensor, source_Q, source_T)
        # pred_Q, pred_T = model(target_img, source_img, source_Q, source_T)
        pdb.set_trace()
        success_time = round(time.time() - start_time, 4)   # 计算成功所需时间

        # estimate_pose = estimate_pose.cpu().detach().numpy().squeeze(0)
        # source_pose_ = source_pose_.cpu().detach().numpy().squeeze(0)
        # pred_R = pred_R.cpu().detach().numpy().squeeze(0)
        # pred_T = pred_T.cpu().detach().numpy().squeeze(0)
        # source_Q = source_Q.cpu().detach().numpy().squeeze(0)
        # source_T = source_T.cpu().detach().numpy().squeeze(0)

        # estimate_pose = pose_transform(estimate_pose)
        # source_pose_ = pose_transform(source_pose_)

        rot_error, translation_error=cal_campose_error(pred_Q, target_Q , pred_T, target_T)
        pdb.set_trace()
        # print('Rotation error: ', rot_error)
        # print('Translation error: ', translation_error)
        # print('-----------------------------------')

        # 判断是否估计成功
        if rot_error < 5 and translation_error < 0.05:
            # print('pred_R: ', pred_R)
            # print('source_R_: ', source_R_)
            # print('pred_T: ', pred_T)
            # print('source_T_: ', source_T_)c
            print('Rotation error: ', rot_error)
            print('Translation error: ', translation_error)
            print('-----------------------------------')
            print('Pose estimation succeeded!')
            print(f"Time taken to estimate pose: {success_time:.4f} seconds")
        else:
            print('Rotation error: ', rot_error)
            print('Translation error: ', translation_error)
            print('Pose estimation failed!')
            print(f"Time taken to estimate pose: {success_time:.4f} seconds")

    return pred_Q, pred_T

def generate_both_directions_fixed_distance_torch(T, min_dist=0.1, max_dist=0.2, num_samples=3):
    # 将 T 转换为 torch 张量并归一化
    T = torch.tensor(T, dtype=torch.float32)
    direction = T / T.norm()
    
    # 用于存储前方向和反方向的平移向量
    forward_translations = []
    backward_translations = []
    
    for _ in range(num_samples):
        # 生成一个在给定范围内的随机距离
        distance = torch.rand(1).item() * (max_dist - min_dist) + min_dist
        
        # 计算前方向量和反方向量
        T_forward = T + direction * distance
        T_backward = T - direction * distance
        
        forward_translations.append(T_forward)
        backward_translations.append(T_backward)

    forward_translations = torch.cat(forward_translations, dim=0)
    backward_translations = torch.cat(backward_translations, dim=0)

    vector = torch.cat((forward_translations, backward_translations), dim=0)

    return vector

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

if __name__ == "__main__":
    # print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))
    # Set up command line argument parser
    parser = ArgumentParser(description="Camera pose estimation parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    icommaparams = iComMaParams(parser)
    # parser.add_argument("--data_path", default='/home/whao/3D_GS/Feed-forward_iGuassion/data/valdata_synthetic', type=str,help="data path")
    # parser.add_argument("--transformsfile", default='pose.json', type=str,help="data path")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_path", default='output', type=str,help="output path")
    parser.add_argument("--obs_img_index", default=0, type=int)
    parser.add_argument("--delta", default="[30,10,5,0.1,0.2,0.3]", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)

    # device_id = 7  # 这里的 ID 是相对于 CUDA_VISIBLE_DEVICES 的
    # args.data_device = f'cuda:{device_id}'  # 将 cuda 映射到指定的 GPU
    args.data_device = torch.device('cuda:0') 
    
    # # Initialize system state (RNG)
    # safe_state(args.quiet)

    # makedirs(args.output_path, exist_ok=True)
    
    # # load gaussians
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    Scene(gaussians)
    # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    ''' dataset '''
    Dataset = getattr(configs, "test_dataset", None)
    if Dataset is None:
        Dataset = configs.test_dataset
    test_loader = DataLoader(Dataset(), **configs.test_loader_args, pin_memory=True)

    # target_img 源图片
    for idx, (target_img, source_img, source_T, target_T, source_Q, target_Q) in enumerate(test_loader):
        pred_Q, pred_T = val(target_img, source_img, source_Q, source_T, target_Q, target_T)
        estimate_img = rendering(gaussians, pred_Q, pred_T, target_Q, target_T)

    # image_dir = "../saved_images"
    # txt_file = "./saved_txt/pose_info.txt"
    # os.makedirs(image_dir, exist_ok=True)

    # # 预定义转换器，用于将 tensor 转换为图片
    # to_pil = T.ToPILImage()

    # 打开 txt 文件进行写入
    # with open(txt_file, 'w') as f:
    #     for idx, (target_img, source_img, source_T, target_T, source_Q, target_Q) in enumerate(test_loader):
    #         # 调用模型进行预测
            

    #         vector = generate_both_directions_fixed_distance_torch(pred_T)
    #         rotation_vector = pred_Q.expand(vector.size(0)+1, -1)
    #         combined_vectors = torch.cat((pred_T, vector), dim=0)
            
    #         # 渲染估计图像
    #         # estimate_img_1 = rendering(gaussians, pred_Q, pred_T, source_Q, source_T)
    #         for i in range(len(rotation_vector)):
    #             estimate_img = rendering(gaussians, rotation_vector[[i], :], combined_vectors[[i], :], source_Q, source_T)

    #             estimate_img = estimate_img.squeeze(0)
    #             estimate_img_filename = os.path.join(image_dir, f"estimate_img_{i}.png")

    #             rgb = estimate_img.clone().permute(1, 2, 0).cpu().detach().numpy()
    #             # 将渲染图像转换为8位像素格式
    #             rgb8 = to8b(rgb)
    #             imageio.imwrite(estimate_img_filename, rgb8)

    #         poses_es = torch.cat((combined_vectors, rotation_vector), dim=1)


    #         # 将 target_img 和 estimate_img 从 torch.Tensor 转换为 PIL 图像
    #         source_img_pil = to_pil(source_img.squeeze(0))  # 去除批次维度
    #         # estimate_img = estimate_img.squeeze(0) # 去除批次维度
            
    #         # 保存 target_img 和 estimate_img
    #         source_img_filename = os.path.join(image_dir, f"source_img.png")
    #         # estimate_img_filename = os.path.join(image_dir, f"estimate_img_{idx}.png")
    #         # rgb = estimate_img.clone().permute(1, 2, 0).cpu().detach().numpy()
    #         # # 将渲染图像转换为8位像素格式
    #         # rgb8 = to8b(rgb)
    #         # imageio.imwrite(estimate_img_filename, rgb8)
            
    #         source_img_pil.save(source_img_filename)
    #         # estimate_img_pil.save(estimate_img_filename)


    #         # pose
    #         poses_inverse=[]
    #         for i in range(len(poses_es)):
    #             matrix_t_minus = pose_to_matrix(*poses_es[0].cpu().detach())
    #             matrix_t_minus_i = pose_to_matrix(*poses_es[i].cpu().detach())

    #             reference_matrix = torch.inverse(matrix_t_minus)

    #             new_matrix_t_minus = reference_matrix @ matrix_t_minus
    #             new_matrix_t_minus_i = reference_matrix @ matrix_t_minus_i

    #             pose_t_i = torch.tensor(matrix_to_pose(new_matrix_t_minus_i)).to('cuda')
    #             poses_inverse.append(pose_t_i)
    #         pdb.set_trace()


    #         pred_Q = pred_Q / torch.norm(pred_Q, dim=1, keepdim=True)  # 归一化每个四元数
    #         source_Q = source_Q / torch.norm(source_Q, dim=1, keepdim=True)  # 归一化每个四元数

    #         pred_pose = pred_T.squeeze(0).tolist() + pred_Q.squeeze(0).tolist()
    #         source_pose = source_T.squeeze(0).tolist() + source_Q.squeeze(0).tolist()
    #         pose_str = str(pred_pose) + " " + str(source_pose)
    #         f.write(f"{pose_str}\n")

    #         if idx == 25:
    #             break
            
            # 将位姿信息写入 txt 文件
            # 这里将每个批次的 pred_Q, pred_T, source_Q, source_T 数据按行写入 txt 文件
            # f.write(f"pred_Q_{idx}: {[pred_Q]}\n")
            # f.write(f"pred_T_{idx}: {[pred_T]}\n")
            # f.write(f"source_Q_{idx}: {[source_Q]}\n")
            # f.write(f"source_T_{idx}: {[source_T]}\n")
            # f.write("\n")  # 空行分隔每个批次的位姿数据 
    
    




