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


''' data split '''
# train_gt_paths = '/data1/whao_dataset/OpenLane/OpenLane_1.2/openlane1.2/OpenLane/lane3d_1000_v1.1/training'
# train_image_paths = '/data1/whao_dataset/OpenLane/OpenLane_1.2/openlane1.2/OpenLane/images/training'
# val_gt_paths = '/data1/whao_dataset/OpenLane/OpenLane_1.2/openlane1.2/OpenLane/lane3d_1000_v1.1/validation'
# val_image_paths = '/data1/whao_dataset/OpenLane/OpenLane_1.2/openlane1.2/OpenLane/images/validation'

model_save_path = "/data1/whao_model/pose_estimate/whole_process/eight/step--1--1"

''' loader '''
# x_range = (3, 103)
# y_range = (-12, 12)
# meter_per_pixel = 0.5 # grid size
# bev_shape = (int((x_range[1] - x_range[0]) / meter_per_pixel),int((y_range[1] - y_range[0]) / meter_per_pixel))

train_loader_args = dict(
    batch_size=1,
    num_workers=4,
    shuffle=True
)
val_loader_args = dict(
    batch_size=1,
    num_workers=4,
    shuffle=True
)
test_loader_args = dict(
    batch_size=1,
    num_workers=4,
    shuffle=True
)

'''hparams'''
hparams = argparse.Namespace(
    input_height=400,
    input_width=400,
    frnt_rng=1,
    n_channel=128,  
    n_layers=8,  

    # base_dirs_train = ['../data/nerf_synthetic/lego', '../data/nerf_synthetic/chair', '../data/nerf_synthetic/drums', '../data/nerf_synthetic/ficus', 
    #              '../data/nerf_synthetic/hotdog', '../data/nerf_synthetic/lego', '../data/nerf_synthetic/materials', '../data/nerf_synthetic/mic', '../data/nerf_synthetic/ship'],
    base_dirs_train = ['/home/whao/pose_eatimate/data/nerf_synthetic/chair', 
                       '/home/whao/pose_eatimate/data/nerf_synthetic/drums', 
                       '/home/whao/pose_eatimate/data/nerf_synthetic/ficus',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/hotdog',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/lego',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/materials',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/mic',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/ship',
                       ],

    base_dirs_val = ['/home/whao/pose_eatimate/data/nerf_synthetic/chair', 
                       '/home/whao/pose_eatimate/data/nerf_synthetic/drums', 
                       '/home/whao/pose_eatimate/data/nerf_synthetic/ficus',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/hotdog',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/lego',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/materials',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/mic',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/ship',
                       ],

    base_dirs_test = ['/home/whao/pose_eatimate/data/nerf_synthetic/chair', 
                       '/home/whao/pose_eatimate/data/nerf_synthetic/drums', 
                       '/home/whao/pose_eatimate/data/nerf_synthetic/ficus',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/hotdog',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/lego',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/materials',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/mic',
                       '/home/whao/pose_eatimate/data/nerf_synthetic/ship',
                       ],
    half_res=True,
    testskip=1,
    n_fremes=7
)

# ''' virtual camera config '''
# vc_config = {}
# vc_config['use_virtual_camera'] = True
# vc_config['vc_intrinsic'] = np.array([[2081.5212033927246, 0.0, 934.7111248349433],
#                                     [0.0, 2081.5212033927246, 646.3389987785433],
#                                     [0.0, 0.0, 1.0]])
# vc_config['vc_extrinsics'] = np.array(
#         [[-0.002122161262459438, 0.010697496358766389, 0.9999405282331697, 1.5441039498273286],
#             [-0.9999378331046326, -0.010968621415360667, -0.0020048117763292747, -0.023774034344867204],
#             [0.010946522625388108, -0.9998826195688676, 0.01072010851209982, 2.1157397903843567],
#             [0.0, 0.0, 0.0, 1.0]])
# vc_config['vc_image_shape'] = (1920, 1280)


''' model '''
def model():
    return Pose_Pred(hparams)


''' optimizer '''
epochs = 800
optimizer = AdamW
optimizer_params = dict(
    lr=5e-6, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=1e-2, amsgrad=False
)
# scheduler = ReduceLROnPlateau
# scheduler_params = dict(
#     mode='min', patience=2, factor=0.1, verbose=True
# )
scheduler = CosineAnnealingLR


def train_dataset():
    train_trans = A.Compose([
                    # A.RandomBrightnessContrast(),
                    # A.ColorJitter(p=0.5),
                    # A.Normalize(),
                    # ToTensorV2()
                    ])
    train_data = BlenderDataset(hparams.base_dirs_train, train_trans, hparams.half_res, hparams.testskip, hparams.n_fremes)

    return train_data

def val_dataset():
    train_trans = A.Compose([
                    # A.RandomBrightnessContrast(),
                    # A.ColorJitter(p=0.5),
                    # A.Normalize(),
                    # ToTensorV2()
                    ])
    val_data = BlenderDataset_val(hparams.base_dirs_val, train_trans, hparams.half_res, hparams.testskip, hparams.n_fremes)

    return val_data

def test_dataset():
    train_trans = A.Compose([
                    # A.RandomBrightnessContrast(),
                    # A.ColorJitter(p=0.5),
                    # A.Normalize(),
                    # ToTensorV2()
                    ])
    test_data = BlenderDataset_test(hparams.base_dirs_test, train_trans, hparams.half_res, hparams.testskip, hparams.n_fremes)

    return test_data


# def val_dataset():
#     val_data = OpenLane_dataset_with_offset_val(val_image_paths,val_gt_paths,
#                                                 trans_image,vc_config)
#     return val_data



