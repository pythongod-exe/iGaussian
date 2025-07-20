from PIL import Image
from pathlib import Path
import numpy as np
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

data_path = '/home/whao/3D_GS/nerf/nerf-pytorch-master/data/nerf_synthetic/lego'
transformsfile = '/home/whao/3D_GS/nerf/nerf-pytorch-master/data/nerf_synthetic/lego/transforms_train.json'

class CameraInfo(NamedTuple):
    pose: np.array
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

def camera_pose_estimation(gaussians:GaussianModel, cam_infos, args):
    # background
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # generate camera_info
    test_info = get_pose_estimation_input(cam_infos, ast.literal_eval(args.delta))

    # start pose & gt pose
    gt_pose_c2w=test_info.gt_pose_c2w
    start_pose_w2c=test_info.start_pose_w2c.cuda()
    
    # # query_image for comparing 
    # query_image = icomma_info.query_image.cuda()


    # initialize camera pose object
    camera_pose = Camera_Pose(start_pose_w2c,FoVx=test_info.FoVx,FoVy=test_info.FoVy,
                            image_width=test_info.image_width,image_height=test_info.image_height)
    camera_pose.cuda()

    rendering = render(camera_pose, gaussians, background, compute_grad_cov2d=True)["render"]

    return cam_infos, rendering

def readCamerasFromTransforms(path, transformsfile, white_background=False, extension=".png"):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents["camera_angle_x"]
        fovx = 0.6911112070083618

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(pose=c2w, uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

