#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from gaussian.utils.image_utils import rgb2loftrgray
from utils.metrics import estimate_pose
import numpy as np
import pdb
import time

def lofter_matching_prior_ransac(q_img,r_img,matcher,threshold,min_num_pointsm,translation_scale=None,relative_gt=None):
    q_img_gray=rgb2loftrgray(q_img) # 将rgb转换为loftr所需的灰度图像
    r_img_gray = rgb2loftrgray(r_img) 
    batch = {'image0':q_img_gray, 'image1':r_img_gray}
    matcher(batch)
    # mkpts0 = batch['mkpts0_f'] # 查询图像中的匹配点坐标
    # mkpts1 = batch['mkpts1_f'] # 渲染图像中的匹配点坐标
    # mconf = batch['mconf'] # 匹配点的置信度

    # # extra matching Information
    # m_bids = batch['m_bids']
    # pdb.set_trace()

    # matching Information
    pixel_thr = 0.5
    conf = 0.99999

    m_bids = batch['m_bids'] # 匹配点属于的批次索引
    pts0 = batch['mkpts0_f']
    pts1 = batch['mkpts1_f']

    # get intrinsics
    K_0, K_1 = get_intrinsics()
    K0 = K_0
    K1 = K_1

    pred_rt = None
    pred_e = None
    # batch['translation_scale']=translation_scale
    # batch['priorRT']=torch.tensor(relative_gt).float()
    # priorRT=batch['priorRT']

    for bs in range(K0.shape[0]):
        mask = m_bids == bs

        ret, num_correspondences_after_ransac, inliers_best_tight, inliers_best_ultra_tight = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], 
                            pixel_thr, conf=conf, 
                            translation_scale=translation_scale, 
                            solver='prior_ransac', 
                            priorRT=relative_gt) # ret -> R t mask E
        
        if ret is not None:
            pred_rt = torch.cat([ret[0],ret[1].unsqueeze(1)],axis=1)
            pred_e = ret[3]
        else:
            pred_rt = torch.cat([torch.eye(3),torch.zeros([3,1])],axis=1).cuda()
            pred_e = torch.eye(3).cuda()

    return pred_rt

def get_intrinsics(fx=1111.111, fy=1111.111, cx=400.0, cy=400.0):
        K = [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
        K = np.array(K)
        K = torch.from_numpy(K.astype(np.double)).unsqueeze(0).cuda()
        return K, K
        
