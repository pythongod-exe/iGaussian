import numpy as np
import torch
import os
from torch import nn
from utils.config_util import load_config_module
import cv2
import imageio
from argparse import ArgumentParser
from gaussian.arguments import ModelParams, PipelineParams,iComMaParams, get_combined_args
from gaussian.gaussian_renderer import GaussianModel
# from gaussian.scene.__init__ import Scene
from rendering.train_sence import Scene
from gaussian.utils.icomma_helper import load_LoFTR, get_pose_estimation_input, get_pose_estimation_input_copy, get_pose_estimation_input_1
from gaussian.run_source_imgs import camera_pose_estimation
from loader.loader_helper import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix
from gaussian.utils.image_utils import to8b
from gaussian.utils.graphics_utils import getWorld2View2, getProjectionMatrix, se3_to_SE3
import time

import pdb

def normalize_quaternion(q):
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

def generate_image(pred_T, pred_Q, gaussians, background):
    estimate_imgs = []
    for i in range (pred_T.shape[0]):
        pred_T_i = pred_T[i]
        pred_Q_i = pred_Q[i]
        T = pred_T_i.squeeze(0).detach().cpu().numpy()
        Q = pred_Q_i.squeeze(0).detach().cpu().numpy()
        Q = normalize_quaternion(Q)
        R = quaternion_to_rotation_matrix(Q)

        pose_c2w = RT2pose(R, T)

        # obs_view_0 = getTestCameras(pose_c2w)[0]
        pose_c2w[:3, 1:3] *= -1
        
        w2c = np.linalg.inv(pose_c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        # scene = Scene(gaussians) 

        cam_infos_0 = get_pose_estimation_input_1(R, T)

        estimate_img, start_pose_w2c = camera_pose_estimation(gaussians,background,cam_infos_0) # [3, 800, 800]\

        estimate_imgs.append(estimate_img)

    return torch.stack(estimate_imgs), start_pose_w2c

def rendering(gaussians, pred_Q, pred_T, source_Q, source_T):
    # # Set up command line argument parser
    # parser = ArgumentParser(description="Camera pose estimation parameters")
    # pipeline = PipelineParams(parser)
    
    # # load gaussians
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # load_iteration=-1 # -1
    estimate_imgs,start_pose_w2c_pred = generate_image(pred_T, pred_Q, gaussians, background)
    source_imgs,start_pose_w2c_gt = generate_image(source_T, source_Q, gaussians, background)

    return source_imgs, estimate_imgs, start_pose_w2c_pred,start_pose_w2c_gt

def rendering_iter(gaussians, translate_vectors, rotation_vectors):
    # # Set up command line argument parser
    # parser = ArgumentParser(description="Camera pose estimation parameters")
    # pipeline = PipelineParams(parser)
    
    # # load gaussians
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # load_iteration=-1 # -1
    gaussian_images,_ = generate_image(translate_vectors, rotation_vectors, gaussians, background)

    return gaussian_images

    # with torch.no_grad():
    #     # 获取渲染图像，并转换为CPU上的NumPy数组
    #     estimate_img = estimate_imgs[0].squeeze(0)
    #     rgb = estimate_img.clone().permute(1, 2, 0).cpu().detach().numpy()
    #     rgb8 = to8b(rgb)
    #     filename = "/home/whao/pose_eatimate/Feed-forward_iGuassion-weight/test_outputs/test_estimate_img.png"
    #     imageio.imwrite(filename, rgb8)

    #     source_img = source_imgs[0].squeeze(0)
    #     rgb = source_img.clone().permute(1, 2, 0).cpu().detach().numpy()
    #     rgb8 = to8b(rgb)
    #     filename = "/home/whao/pose_eatimate/Feed-forward_iGuassion-weight/test_outputs/test_source_img.png"
    #     imageio.imwrite(filename, rgb8)

    #     rgb = estimate_imgs[0].clone().permute(1, 2, 0).cpu().detach().numpy()
    #     rgb8 = to8b(rgb)
    #     ref = to8b(source_imgs[0].permute(1, 2, 0).cpu().detach().numpy())
    #     filename = "/home/whao/pose_eatimate/Feed-forward_iGuassion-weight/test_outputs/test_0.png"
    #     dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
    #     imageio.imwrite(filename, dst)
    #     pdb.set_trace()

    # return estimate_imgs
