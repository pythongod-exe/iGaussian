import sys
sys.path.append('/home/whao/pose_eatimate/Feed-forward_iGuassion-weight-matching-7-pose')
import time
import torch
from torch.utils.data import Dataset,DataLoader
from models.util.load_model import load_model
from utils.config_util import load_config_module
from utils.calculate_error_utils_test import cal_campose_error,calculate_errors_4x4
import time
import pdb
import imageio
from argparse import ArgumentParser
from gaussian.arguments import ModelParams, PipelineParams,iComMaParams, get_combined_args
from gaussian.gaussian_renderer import GaussianModel
# from gaussian.scene import Scene
from rendering.train_sence import Scene
from gaussian.utils.icomma_helper import load_LoFTR, get_pose_estimation_input, get_pose_estimation_input_copy, get_pose_estimation_input_1
from gaussian.utils.image_utils import to8b
# from rendering.rendering_estimate_pose import rendering
from rendering.Guassion_render_train import rendering, rendering_iter
from scipy.spatial.transform import Rotation as R
from utils.matching_utils import lofter_matching_prior_ransac,get_intrinsics
from utils.gaussian_sample import generate_poses
from utils.relative import relative_pose
from utils.calculate_pose_loss import PoseLossCalculator

from PIL import Image
import torch

# model_path = '/data1/whao_model/feed_forward_gaussian/best_model.pth' #model path of verification
model_path = '/data1/whao_model/pose_estimate/whole_process/1/model/ep119.pth' #model path of verification

''' parameter from config '''
config_file = './config_blender.py'
configs = load_config_module(config_file)

def val(target_imgs, source_imgs, source_poses, target_pose, model_net):
    target_imgs = target_imgs.cuda()
    source_imgs = source_imgs.cuda()
    source_poses = source_poses.cuda()
    target_pose = target_pose.cuda()

    pred_pose = model_net(target_imgs, source_imgs, source_poses)
    rot_error, translation_error=cal_campose_error(pred_pose, target_pose)
    pred_pose[0, 3:] = pred_pose[0, 3:] / torch.norm(pred_pose[0, 3:])
    print("---rot_error---",rot_error)
    print("---translation_error---",translation_error)
    pdb.set_trace()

    return pred_pose

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Camera pose estimation parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    icommaparams = iComMaParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_path", default='output', type=str,help="output path")
    parser.add_argument("--obs_img_index", default=0, type=int)
    parser.add_argument("--delta", default="[30,10,5,0.1,0.2,0.3]", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)

    args.data_device = torch.device('cuda:0') 

    # load rela_pose model
    model_net = configs.model()
    model_net = load_model(model_net, model_path)
    print(model_path)
    model_net.cuda()
    # model_net.train()

    # load LoFTR_model
    LoFTR_model=load_LoFTR(icommaparams.LoFTR_ckpt_path,icommaparams.LoFTR_temp_bug_fix)
    
    # # load gaussians
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    Scene(gaussians)

    # # ''' dataset '''
    Dataset = getattr(configs, "test_dataset", None)
    if Dataset is None:
        Dataset = configs.test_dataset
    test_loader = DataLoader(Dataset(), **configs.test_loader_args, pin_memory=True)

    # # get matching picture
    for idx, (target_imgs, source_imgs, source_poses, target_pose) in enumerate(test_loader):
        # step1 
        start_time = time.time()
        pred_pose = val(target_imgs, source_imgs, source_poses, target_pose, model_net) # [1, 4, 3, 400, 400] [1, 4, 3, 400, 400] [1, 4, 4] [1, 4, 3] [1, 4] [1, 3]
        pdb.set_trace()

        # step2
        translate_vectors, rotation_vectors = generate_poses(pred_Q_1, pred_T_1)
        gaussian_images = rendering_iter(gaussians, translate_vectors, rotation_vectors)
        pdb.set_trace()
        pred_Q_2, pred_T_2 = val(target_imgs, gaussian_images.unsqueeze(0), rotation_vectors.unsqueeze(0), translate_vectors.unsqueeze(0), target_Q, target_T, model_net) # [1, 4, 3, 400, 400] [1, 4, 3, 800, 800] torch.Size([1, 4, 4]) 
        target_gt_img, target_estimate_img, start_pose_w2c_pred, start_pose_w2c_gt = rendering(gaussians, pred_Q_1, pred_T_1, target_Q, target_T)

        # step3 start->gt
        translation_scale = torch.linalg.norm(pred_T_1 - pred_T_2)*1
        # translation_scale = None
        target_gt_img = target_gt_img.squeeze(0)
        target_estimate_img = target_estimate_img.squeeze(0)
        relative_gt=relative_pose(start_pose_w2c_pred.cpu().detach(),start_pose_w2c_gt.cpu().detach())

        pred_rt = lofter_matching_prior_ransac(target_estimate_img,target_gt_img,LoFTR_model,icommaparams.confidence_threshold_LoFTR,icommaparams.min_matching_points,translation_scale,relative_gt) # 0.0718529224395752s

        translation_error, rotation_error = calculate_errors_4x4(relative_gt, pred_rt.cpu().numpy())
        success_time = time.time() - start_time
        print("---success_time---", success_time)
        pdb.set_trace()



    




