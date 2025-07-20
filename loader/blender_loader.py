import os
import torch
import numpy as np
import imageio 
import json
import cv2
import random
from torch.utils.data import Dataset
import pdb
from PIL import Image
from gaussian.utils.general_utils import PILtoTorch
from gaussian.utils.image_utils import to8b
from loader.loader_helper import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix, normalize_quaternion

class  BlenderDataset(Dataset):
    def __init__(self, base_dirs, data_trans, half_res=False, testskip=1, n_fremes=4):
        """
        base_dirs: list of base directories for each scene, e.g., ['./data/nerf_synthetic/lego', './data/nerf_synthetic/chair']
        half_res: whether to load images at half resolution
        testskip: skip interval for test images
        """
        self.base_dirs = base_dirs
        self.half_res = half_res
        self.testskip = testskip
        self.trans_image = data_trans
        self.n_fremes = n_fremes
        self.scenes_data = self._load_all_scenes()
        self.scene_names = list(self.scenes_data.keys())
        self.length_per_scene = {scene: len(data['imgs']) for scene, data in self.scenes_data.items()}
        self.total_length = sum(self.length_per_scene.values())

    def _load_all_scenes(self):
        scenes_data = {}
        for base_dir in self.base_dirs:
            scene_name = os.path.basename(base_dir)
            imgs, poses = self._load_data(base_dir)
            scenes_data[scene_name] = {'imgs': imgs, 'poses': poses}
        return scenes_data
    
    def _load_data(self, basedir):
        # splits = ['train', 'val', 'test']
        splits = ['train', 'val', 'test']
        metas = {}
        for s in splits:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        all_imgs = [] 
        all_T_matrixs = []
        all_Q_matrixs = []
        for s in splits:
            meta = metas[s]
            imgs = []
            T_matrixs = []
            Q_matrixs = []
            
            for frame in meta['frames']:
            # for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir, frame['file_path'] + '.png')

                c2w = np.array(frame["transform_matrix"])
                R = c2w[:3,:3]
                T = c2w[:3, 3]
                quaternion = rotation_matrix_to_quaternion(R) # wxyz
                quaternion = normalize_quaternion(quaternion)

                T_matrixs.append(np.array(T))
                Q_matrixs.append(np.array(quaternion))

                image_path = os.path.join(fname)
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))

                bg = np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])

                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

                resolution = 1
                orig_w, orig_h = image.size
                scale = 1
                resolution = (int(orig_w / scale), int(orig_h / scale))

                resized_image_rgb = PILtoTorch(image, resolution)
                gt_image = resized_image_rgb[:3, ...]
                
                # to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
                # rgb = gt_image.clone().permute(1, 2, 0).cpu().detach().numpy()
                # rgb8 = to8b(rgb)
                # filename = "/home/whao/pose_eatimate/Feed-forward_iGuassion-weight/test_outputs/r_0.png"
                # imageio.imwrite(filename, rgb8)
                # pdb.set_trace()
                
                gt_image = gt_image.permute(1, 2, 0).numpy()
                transformed = self.trans_image(image=gt_image)
                # image = transformed["image"]
                gt_image = transformed["image"]
                gt_image = gt_image.transpose((2, 0, 1))
                gt_image = np.expand_dims(gt_image, axis=0)

                all_imgs.append(gt_image)

            Q_matrixs = np.array(Q_matrixs).astype(np.float32)
            T_matrixs = np.array(T_matrixs).astype(np.float32)
            all_T_matrixs.append(T_matrixs)
            all_Q_matrixs.append(Q_matrixs)
            T_matrixs = np.concatenate(all_T_matrixs, 0)
            Q_matrixs = np.concatenate(all_Q_matrixs, 0)
            poses = np.concatenate((T_matrixs, Q_matrixs), axis=1)

        return np.concatenate(all_imgs, 0), poses

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # 根据全局idx找到对应的场景和场景内的局部idx
        current_scene_idx = 0
        while idx >= self.length_per_scene[self.scene_names[current_scene_idx]]:
            idx -= self.length_per_scene[self.scene_names[current_scene_idx]]
            current_scene_idx += 1
        scene_name = self.scene_names[current_scene_idx]
        scene_data = self.scenes_data[scene_name]

        # 获取当前场景内的图像和姿态
        target_img = scene_data['imgs'][idx]
        target_pose = scene_data['poses'][idx]

        # remaining_indices = list(range(self.length_per_scene[scene_name]))
        # remaining_indices.remove(idx)
        # source_indices = random.sample(remaining_indices, self.n_fremes)  # 随机选取 4 个索引
        source_indices = [0, 2, 4, 13, 24, 46, 48]
        # source_indices = [0, 2, 4, 13, 24]

        # Step 4: 提取 4 个源图像和姿态
        source_imgs = []
        source_poses = []
        for source_idx in source_indices:
            source_img = scene_data['imgs'][source_idx]

            source_pose = scene_data['poses'][source_idx]

            source_imgs.append(torch.from_numpy(source_img).float())
            source_poses.append(torch.from_numpy(source_pose).float())

        # rgb = torch.tensor(source_imgs[6]).permute(1, 2, 0).cpu().detach().numpy()
        # rgb8 = to8b(rgb)
        # filename = "/home/whao/pose_eatimate/Feed-forward_iGuassion-weight-matching/test_img.png"
        # imageio.imwrite(filename, rgb8)
        # pdb.set_trace()

        source_imgs = torch.stack(source_imgs)
        source_poses = torch.stack(source_poses)

        # Step 5: 转换目标图像和姿态为张量
        target_img = torch.from_numpy(target_img).float()
        target_imgs = target_img.unsqueeze(0).repeat(self.n_fremes, 1, 1, 1)
        target_pose = torch.from_numpy(target_pose).float()
        # target_Ts = target_T.unsqueeze(0).repeat(self.n_fremes, 1) 
        # target_Qs = target_Q.unsqueeze(0).repeat(self.n_fremes, 1) 

        return target_imgs, source_imgs, source_poses, target_pose


class  BlenderDataset_val(Dataset):
    def __init__(self, base_dirs, data_trans, half_res=False, testskip=1, n_fremes=4):
        """
        base_dirs: list of base directories for each scene, e.g., ['./data/nerf_synthetic/lego', './data/nerf_synthetic/chair']
        half_res: whether to load images at half resolution
        testskip: skip interval for test images
        """
        self.base_dirs = base_dirs
        self.half_res = half_res
        self.testskip = testskip
        self.trans_image = data_trans
        self.n_fremes = n_fremes
        self.scenes_data = self._load_all_scenes()
        self.scene_names = list(self.scenes_data.keys())
        self.length_per_scene = {scene: len(data['imgs']) for scene, data in self.scenes_data.items()}
        self.total_length = sum(self.length_per_scene.values())

    def _load_all_scenes(self):
        scenes_data = {}
        for base_dir in self.base_dirs:
            scene_name = os.path.basename(base_dir)
            imgs, poses = self._load_data(base_dir)
            scenes_data[scene_name] = {'imgs': imgs, 'poses': poses}
        return scenes_data
    
    def _load_data(self, basedir):
        # splits = ['train', 'val', 'test']
        splits = ['train', 'val', 'test']
        metas = {}
        for s in splits:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        all_imgs = [] 
        all_T_matrixs = []
        all_Q_matrixs = []
        for s in splits:
            meta = metas[s]
            imgs = []
            T_matrixs = []
            Q_matrixs = []
            
            for frame in meta['frames']:
            # for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir, frame['file_path'] + '.png')

                c2w = np.array(frame["transform_matrix"])
                R = c2w[:3,:3]
                T = c2w[:3, 3]
                quaternion = rotation_matrix_to_quaternion(R) # wxyz
                quaternion = normalize_quaternion(quaternion)

                T_matrixs.append(np.array(T))
                Q_matrixs.append(np.array(quaternion))

                image_path = os.path.join(fname)
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))

                bg = np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])

                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

                resolution = 1
                orig_w, orig_h = image.size
                scale = 1
                resolution = (int(orig_w / scale), int(orig_h / scale))

                resized_image_rgb = PILtoTorch(image, resolution)
                gt_image = resized_image_rgb[:3, ...]
                
                # to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
                # rgb = gt_image.clone().permute(1, 2, 0).cpu().detach().numpy()
                # rgb8 = to8b(rgb)
                # filename = "/home/whao/pose_eatimate/Feed-forward_iGuassion-weight/test_outputs/r_0.png"
                # imageio.imwrite(filename, rgb8)
                # pdb.set_trace()
                
                gt_image = gt_image.permute(1, 2, 0).numpy()
                transformed = self.trans_image(image=gt_image)
                # image = transformed["image"]
                gt_image = transformed["image"]
                gt_image = gt_image.transpose((2, 0, 1))
                gt_image = np.expand_dims(gt_image, axis=0)

                all_imgs.append(gt_image)

            Q_matrixs = np.array(Q_matrixs).astype(np.float32)
            T_matrixs = np.array(T_matrixs).astype(np.float32)
            all_T_matrixs.append(T_matrixs)
            all_Q_matrixs.append(Q_matrixs)
            T_matrixs = np.concatenate(all_T_matrixs, 0)
            Q_matrixs = np.concatenate(all_Q_matrixs, 0)
            poses = np.concatenate((T_matrixs, Q_matrixs), axis=1)

        return np.concatenate(all_imgs, 0), poses

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # 根据全局idx找到对应的场景和场景内的局部idx
        current_scene_idx = 0
        while idx >= self.length_per_scene[self.scene_names[current_scene_idx]]:
            idx -= self.length_per_scene[self.scene_names[current_scene_idx]]
            current_scene_idx += 1

        scene_name = self.scene_names[current_scene_idx]
        scene_data = self.scenes_data[scene_name]

        # 获取当前场景内的图像和姿态
        target_img = scene_data['imgs'][idx]
        target_pose = scene_data['poses'][idx]

        # remaining_indices = list(range(self.length_per_scene[scene_name]))
        # remaining_indices.remove(idx)
        # source_indices = random.sample(remaining_indices, self.n_fremes)  # 随机选取 4 个索引
        source_indices = [0, 2, 4, 13, 24, 46, 48]
        # source_indices = [0, 2, 4, 13, 24]

        # Step 4: 提取 4 个源图像和姿态
        source_imgs = []
        source_poses = []
        for source_idx in source_indices:
            source_img = scene_data['imgs'][source_idx]

            source_pose = scene_data['poses'][source_idx]

            source_imgs.append(torch.from_numpy(source_img).float())
            source_poses.append(torch.from_numpy(source_pose).float())

        # rgb = torch.tensor(source_imgs[6]).permute(1, 2, 0).cpu().detach().numpy()
        # rgb8 = to8b(rgb)
        # filename = "/home/whao/pose_eatimate/Feed-forward_iGuassion-weight-matching/test_img.png"
        # imageio.imwrite(filename, rgb8)
        # pdb.set_trace()

        source_imgs = torch.stack(source_imgs)
        source_poses = torch.stack(source_poses)

        # Step 5: 转换目标图像和姿态为张量
        target_img = torch.from_numpy(target_img).float()
        target_imgs = target_img.unsqueeze(0).repeat(self.n_fremes, 1, 1, 1)
        target_pose = torch.from_numpy(target_pose).float()
        # target_Ts = target_T.unsqueeze(0).repeat(self.n_fremes, 1) 
        # target_Qs = target_Q.unsqueeze(0).repeat(self.n_fremes, 1) 

        return target_imgs, source_imgs, source_poses, target_pose
    

class  BlenderDataset_test(Dataset):
    def __init__(self, base_dirs, data_trans, half_res=False, testskip=1, n_fremes=4):
        """
        base_dirs: list of base directories for each scene, e.g., ['./data/nerf_synthetic/lego', './data/nerf_synthetic/chair']
        half_res: whether to load images at half resolution
        testskip: skip interval for test images
        """
        self.base_dirs = base_dirs
        self.half_res = half_res
        self.testskip = testskip
        self.trans_image = data_trans
        self.n_fremes = n_fremes
        self.scenes_data = self._load_all_scenes()
        self.scene_names = list(self.scenes_data.keys())
        self.length_per_scene = {scene: len(data['imgs']) for scene, data in self.scenes_data.items()}
        self.total_length = sum(self.length_per_scene.values())

    def _load_all_scenes(self):
        scenes_data = {}
        for base_dir in self.base_dirs:
            scene_name = os.path.basename(base_dir)
            imgs, poses = self._load_data(base_dir)
            scenes_data[scene_name] = {'imgs': imgs, 'poses': poses}
        return scenes_data
    
    def _load_data(self, basedir):
        # splits = ['train', 'val', 'test']
        splits = ['train', 'val', 'test']
        metas = {}
        for s in splits:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        all_imgs = [] 
        all_T_matrixs = []
        all_Q_matrixs = []
        for s in splits:
            meta = metas[s]
            imgs = []
            T_matrixs = []
            Q_matrixs = []
            
            for frame in meta['frames']:
            # for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir, frame['file_path'] + '.png')

                c2w = np.array(frame["transform_matrix"])
                R = c2w[:3,:3]
                T = c2w[:3, 3]
                quaternion = rotation_matrix_to_quaternion(R) # wxyz
                quaternion = normalize_quaternion(quaternion)

                T_matrixs.append(np.array(T))
                Q_matrixs.append(np.array(quaternion))

                image_path = os.path.join(fname)
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))

                bg = np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])

                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

                resolution = 1
                orig_w, orig_h = image.size
                scale = 1
                resolution = (int(orig_w / scale), int(orig_h / scale))

                resized_image_rgb = PILtoTorch(image, resolution)
                gt_image = resized_image_rgb[:3, ...]
                
                # to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
                # rgb = gt_image.clone().permute(1, 2, 0).cpu().detach().numpy()
                # rgb8 = to8b(rgb)
                # filename = "/home/whao/pose_eatimate/Feed-forward_iGuassion-weight/test_outputs/r_0.png"
                # imageio.imwrite(filename, rgb8)
                # pdb.set_trace()
                
                gt_image = gt_image.permute(1, 2, 0).numpy()
                transformed = self.trans_image(image=gt_image)
                # image = transformed["image"]
                gt_image = transformed["image"]
                gt_image = gt_image.transpose((2, 0, 1))
                gt_image = np.expand_dims(gt_image, axis=0)

                all_imgs.append(gt_image)

            Q_matrixs = np.array(Q_matrixs).astype(np.float32)
            T_matrixs = np.array(T_matrixs).astype(np.float32)
            all_T_matrixs.append(T_matrixs)
            all_Q_matrixs.append(Q_matrixs)
            T_matrixs = np.concatenate(all_T_matrixs, 0)
            Q_matrixs = np.concatenate(all_Q_matrixs, 0)
            poses = np.concatenate((T_matrixs, Q_matrixs), axis=1)

        return np.concatenate(all_imgs, 0), poses

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # 根据全局idx找到对应的场景和场景内的局部idx
        current_scene_idx = 0
        while idx >= self.length_per_scene[self.scene_names[current_scene_idx]]:
            idx -= self.length_per_scene[self.scene_names[current_scene_idx]]
            current_scene_idx += 1

        scene_name = self.scene_names[current_scene_idx]
        scene_data = self.scenes_data[scene_name]

        # 获取当前场景内的图像和姿态
        target_img = scene_data['imgs'][idx]
        target_pose = scene_data['poses'][idx]

        # remaining_indices = list(range(self.length_per_scene[scene_name]))
        # remaining_indices.remove(idx)
        # source_indices = random.sample(remaining_indices, self.n_fremes)  # 随机选取 4 个索引
        source_indices = [0, 2, 4, 13, 24, 46, 48]
        # source_indices = [0, 2, 4, 13, 24]

        # Step 4: 提取 4 个源图像和姿态
        source_imgs = []
        source_poses = []
        for source_idx in source_indices:
            source_img = scene_data['imgs'][source_idx]

            source_pose = scene_data['poses'][source_idx]

            source_imgs.append(torch.from_numpy(source_img).float())
            source_poses.append(torch.from_numpy(source_pose).float())

        # rgb = torch.tensor(source_imgs[6]).permute(1, 2, 0).cpu().detach().numpy()
        # rgb8 = to8b(rgb)
        # filename = "/home/whao/pose_eatimate/Feed-forward_iGuassion-weight-matching/test_img.png"
        # imageio.imwrite(filename, rgb8)
        # pdb.set_trace()

        source_imgs = torch.stack(source_imgs)
        source_poses = torch.stack(source_poses)

        # Step 5: 转换目标图像和姿态为张量
        target_img = torch.from_numpy(target_img).float()
        target_imgs = target_img.unsqueeze(0).repeat(self.n_fremes, 1, 1, 1)
        target_pose = torch.from_numpy(target_pose).float()
        # target_Ts = target_T.unsqueeze(0).repeat(self.n_fremes, 1) 
        # target_Qs = target_Q.unsqueeze(0).repeat(self.n_fremes, 1) 

        return target_imgs, source_imgs, source_poses, target_pose