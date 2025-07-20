import sys
sys.path.append('/home/whao/3D_GS/Feed-forward_iGuassion')
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
from gaussian.scene import Scene
from gaussian.utils.icomma_helper import load_LoFTR, get_pose_estimation_input, get_pose_estimation_input_copy, get_pose_estimation_input_1
from gaussian.run_source_imgs import camera_pose_estimation
from gaussian.run_source_imgs import half_pixle_sampling
from gaussian.run import readCamerasFromTransforms
from loader.loader_helper import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix
from PIL import Image
from torchvision import transforms
from gaussian.utils.image_utils import to8b

''' parameter from config '''
config_file = './config_blender.py'
configs = load_config_module(config_file)

def normalize_quaternion(q):
    """
    对四元数进行归一化
    参数:
    q -- 长度为4的四元数 [w, x, y, z]

    返回:
    归一化后的四元数
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("四元数的模长为0，无法进行归一化")
    return q / norm

def RT2pose(R, T):
    pose_matrix = np.eye(4)
    
    # 将旋转矩阵放到齐次变换矩阵的左上角
    pose_matrix[:3, :3] = R
    
    # 将平移向量放到齐次变换矩阵的右上角
    pose_matrix[:3, 3] = T

    return pose_matrix

def generate_image(pred_T, pred_Q, dataset, gaussians, load_iteration, background, pipeline, icommaparams):
    T = pred_T.squeeze(0).detach().cpu().numpy()
    Q = pred_Q.squeeze(0).detach().cpu().numpy()
    Q = normalize_quaternion(Q)
    R = quaternion_to_rotation_matrix(Q)

    pose_c2w = RT2pose(R, T)
    pose_c2w[:3, 1:3] *= -1
    
    w2c = np.linalg.inv(pose_c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    scene = Scene(dataset,gaussians,load_iteration,shuffle=False) # pose, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid,
    # obs_view_0=scene.getTrainCameras()[0]
    # obs_view_1=scene.getTrainCameras()[1]
    obs_view_0=scene.getTestCameras()[0]
    # image_name_1 = obs_view_1.image_name
    # print("image_name_0", image_name_0)
    # print("image_name_1", image_name_1)
    #obs_view=scene.getTrainCameras()[args.obs_img_index]
    # cam_infos_0=get_pose_estimation_input_1(obs_view_0,ast.literal_eval(args.delta))
    # cam_infos_1=get_pose_estimation_input_1(obs_view_1,ast.literal_eval(args.delta))

    obs_view_0.R = R
    obs_view_0.T = T

    cam_infos_0 = get_pose_estimation_input_1(obs_view_0)
    pdb.set_trace()

    # get estimate_pose image
    estimate_img, start_pose_w2c = camera_pose_estimation(gaussians,background,pipeline,icommaparams,cam_infos_0) # [3, 800, 800]\
    
    return estimate_img, start_pose_w2c


def rendering(dataset, gaussians, args, pred_Q, pred_T, source_Q, source_T):
    # # Set up command line argument parser
    parser = ArgumentParser(description="Camera pose estimation parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    icommaparams = iComMaParams(parser)
    # parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--output_path", default='output', type=str,help="output path")
    # parser.add_argument("--obs_img_index", default=0, type=int)
    # parser.add_argument("--delta", default="[30,10,5,0.1,0.2,0.3]", type=str)
    # parser.add_argument("--iteration", default=-1, type=int)
    # args = get_combined_args(parser)

    # args.data_device = torch.device('cuda:0') 
    
    # # Initialize system state (RNG)
    # safe_state(args.quiet)

    # makedirs(args.output_path, exist_ok=True)
    
    # # load gaussians
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # ''' dataset '''
    # Dataset = getattr(configs, "train_dataset", None)
    # if Dataset is None:
    #     Dataset = configs.train_dataset
    # train_loader = DataLoader(Dataset(), **configs.train_loader_args, pin_memory=True)

    # get camera info from Scene
    # Reused 3DGS code to obtain camera information. 
    # You can customize the iComMa_input_info in practical applications.
    # T = pred_T.squeeze(0).detach().cpu().numpy()
    # Q = pred_Q.squeeze(0).detach().cpu().numpy()
    # Q = normalize_quaternion(Q)
    # R = quaternion_to_rotation_matrix(Q)

    # pose_c2w = RT2pose(R, T)
    # pose_c2w[:3, 1:3] *= -1
    
    # w2c = np.linalg.inv(pose_c2w)
    # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    # T = w2c[:3, 3]

    # scene = Scene(dataset,gaussians,load_iteration=args.iteration,shuffle=False) # pose, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid,
    # # obs_view_0=scene.getTrainCameras()[0]
    # # obs_view_1=scene.getTrainCameras()[1]
    # obs_view_0=scene.getTestCameras()[0]
    # # image_name_1 = obs_view_1.image_name
    # # print("image_name_0", image_name_0)
    # # print("image_name_1", image_name_1)
    # #obs_view=scene.getTrainCameras()[args.obs_img_index]
    # # cam_infos_0=get_pose_estimation_input_1(obs_view_0,ast.literal_eval(args.delta))
    # # cam_infos_1=get_pose_estimation_input_1(obs_view_1,ast.literal_eval(args.delta))

    # obs_view_0.R = R
    # obs_view_0.T = T

    # cam_infos_0 = get_pose_estimation_input_1(obs_view_0)

    # # get estimate_pose image
    # estimate_img, cur_pose_c2w = camera_pose_estimation(gaussians,background,pipeline,icommaparams,cam_infos_0) # [3, 800, 800]
    
    load_iteration=args.iteration
    estimate_img = generate_image(pred_T, pred_Q, dataset, gaussians, load_iteration, background, pipeline, icommaparams)
    source_img  = generate_image(source_T, source_Q, dataset, gaussians, load_iteration, background, pipeline, icommaparams)

    with torch.no_grad():
        # 获取渲染图像，并转换为CPU上的NumPy数组
        # estimate_img = estimate_img.squeeze(0)
        # rgb = estimate_img.clone().permute(1, 2, 0).cpu().detach().numpy()
        # # 将渲染图像转换为8位像素格式
        # rgb8 = to8b(rgb)
        # # 构建文件名和保存路径
        # filename = "/home/whao/3D_GS/Feed-forward_iGuassion/outputs/r_0.png"
        # # filename = os.path.join(save_path, f"{image_name_1}.png")
        # # 保存渲染图像
        # imageio.imwrite(filename, rgb8)
        # pdb.set_trace()
        
        rgb = estimate_img.clone().permute(1, 2, 0).cpu().detach().numpy()
        rgb8 = to8b(rgb)
        ref = to8b(source_img.permute(1, 2, 0).cpu().detach().numpy())
        filename = "/home/whao/3D_GS/Feed-forward_iGuassion/outputs/r_test_2.png"
        dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
        imageio.imwrite(filename, dst)
        pdb.set_trace()
        # imgs.append(dst)