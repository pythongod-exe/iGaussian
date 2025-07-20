from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import json
import os
import torch
from gaussian.gaussian_renderer import render
from gaussian.arguments import ModelParams, PipelineParams,iComMaParams, get_combined_args
from gaussian.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from gaussian.utils.icomma_helper import get_pose_estimation_input
from gaussian.gaussian_renderer import GaussianModel
from gaussian.scene.cameras import Camera_Pose
from typing import NamedTuple
import ast
from argparse import ArgumentParser
import pdb
import time

# def camera_pose_estimation(gaussians:GaussianModel, background:torch.tensor, pipeline:PipelineParams, icommaparams:iComMaParams, icomma_info):
#     # start pose & gt pose
#     gt_pose_c2w=icomma_info.gt_pose_c2w
#     start_pose_w2c=icomma_info.start_pose_w2c.cuda()

#     # estimate_pose = estimate_pose.squeeze(0).cpu().detach().numpy()
#     # estimate_pose = estimate_pose.cpu().detach().numpy()
#     # start_pose_w2c=torch.from_numpy(np.linalg.inv(estimate_pose)).float()
#     # start_pose_w2c=torch.tensor(start_pose_w2c).cuda()

#     # initialize camera pose object
#     camera_pose = Camera_Pose(start_pose_w2c,FoVx=icomma_info.FoVx,FoVy=icomma_info.FoVy,
#                             image_width=icomma_info.image_width,image_height=icomma_info.image_height)
#     camera_pose.cuda()

#     rendering = render(camera_pose,gaussians, pipeline, background,compute_grad_cov2d = icommaparams.compute_grad_cov2d)["render"]

#     cur_pose_c2w= camera_pose.current_campose_c2w()

#     return rendering, cur_pose_c2w

def camera_pose_estimation(gaussians:GaussianModel, background:torch.tensor, icomma_info):
    # start pose & gt pose
    gt_pose_c2w=icomma_info.gt_pose_c2w
    start_pose_w2c=icomma_info.start_pose_w2c.cuda()

    # estimate_pose = estimate_pose.squeeze(0).cpu().detach().numpy()
    # estimate_pose = estimate_pose.cpu().detach().numpy()
    # start_pose_w2c=torch.from_numpy(np.linalg.inv(estimate_pose)).float()
    # start_pose_w2c=torch.tensor(start_pose_w2c).cuda()

    # initialize camera pose object
    camera_pose = Camera_Pose(start_pose_w2c,FoVx=icomma_info.FoVx,FoVy=icomma_info.FoVy,
                            image_width=icomma_info.image_width,image_height=icomma_info.image_height)
    
    camera_pose.cuda()

    compute_grad_cov2d = True

    rendering = render(camera_pose,gaussians, background,compute_grad_cov2d)["render"]

    cur_pose_c2w= camera_pose.current_campose_c2w()

    return rendering, start_pose_w2c

def camera_pose_estimation_copy(gaussians:GaussianModel, background:torch.tensor, pipeline:PipelineParams, icommaparams:iComMaParams, icomma_info, output_path, estimate_pose):
    # start pose & gt pose
    gt_pose_c2w=icomma_info.gt_pose_c2w
    # start_pose_w2c=icomma_info.start_pose_w2c.cuda()
    # estimate_pose = estimate_pose.squeeze(0).cpu().detach().numpy()
    start_pose_w2c=torch.from_numpy(np.linalg.inv(estimate_pose)).float()
    start_pose_w2c=torch.tensor(start_pose_w2c).cuda()

    # initialize camera pose object
    camera_pose = Camera_Pose(start_pose_w2c,FoVx=icomma_info.FoVx,FoVy=icomma_info.FoVy,
                            image_width=icomma_info.image_width,image_height=icomma_info.image_height)
    camera_pose.cuda()

    rendering = render(camera_pose,gaussians, pipeline, background,compute_grad_cov2d = icommaparams.compute_grad_cov2d)["render"]

    cur_pose_c2w= camera_pose.current_campose_c2w()

    return rendering

def half_pixle_sampling(image):
    # 将 PyTorch Tensor 转换为 NumPy 数组
    image = image.cpu().detach().numpy()

    # 将图像从 (C, H, W) 转换为 (H, W, C)
    image = np.transpose(image, (1, 2, 0))

    # 获取高度和宽度
    H, W, C = image.shape  # C 是通道数，这里为3
    H_half, W_half = H // 2, W // 2

    # 使用 OpenCV 进行缩放
    image_half_res = cv2.resize(image, (W_half, H_half), interpolation=cv2.INTER_AREA)

    # 将缩放后的图像从 (H, W, C) 转换回 (C, H, W)
    image_half_res = np.transpose(image_half_res, (2, 0, 1))
    image_half_res = torch.from_numpy(image_half_res).cuda()

    return image_half_res
