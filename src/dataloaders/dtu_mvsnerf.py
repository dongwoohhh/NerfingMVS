import os, sys

sys.path.append('../..')

from .data_utils import rectify_inplane_rotation, random_crop, random_flip, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses

import numpy as np
import torch
from torch.utils.data import Dataset
import glob

from utils.io_utils import *


class DTUMVSNeRFDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.args = args
        self.datadir = args.datadir
        list_prefix = args.list_prefix

        self.num_source_views = args.num_source_views
        self.rectify_inplane_rotation = False

        self.depth_H = args.depth_H
        self.depth_W = args.depth_W

        self.mode = mode

        # Init paths.
        all_scenes = [x for x in glob.glob(os.path.join(self.datadir, "*")) if os.path.isdir(x)]
        if mode == "train":
            file_list = os.path.join(self.datadir, list_prefix+"_train.lst")
        elif mode == "val":
            file_list = os.path.join(self.datadir, list_prefix+"_val.lst")
        elif mode == "test":
            file_list = os.path.join(self.datadir, list_prefix+"_test.lst")

        with open(file_list, "r") as f:
            all_scenes = [(x.strip(), os.path.join(self.datadir, x.strip())) for x in f.readlines()]
        self.all_scenes = all_scenes


        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        self.scene_dir = []

        for i, scene in enumerate(all_scenes):
            factor = 1
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(scene[1], load_imgs=False, factor=factor)
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)
            if mode == 'train':
                i_train = np.array(np.arange(int(poses.shape[0])))
                i_render = i_train
            else:
                raise NotImplementedError
                i_test = np.arange(poses.shape[0])[::args.llffhold]
                i_train = np.array([j for j in np.arange(int(poses.shape[0])) if
                                    (j not in i_test and j not in i_test)])
                i_render = i_test

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)
            
            
            for i_image in range(num_render):
                self.scene_dir.append(scene[1])
        
    def __len__(self):
        return len(self.all_scenes)
    
    def __getitem__(self, idx):
        render_pose = self.render_poses[idx]
        depth_range = self.render_depth_range[idx]
        mean_depth = np.mean(depth_range)
        world_center = (render_pose.dot(np.array([[0, 0, mean_depth, 1]]).T)).flatten()[:3]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]

        id_render = -1
        #id_render = train_rgb_files.index(rgb_file)
        subsample_factor = 1
        num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                min(self.num_source_views*subsample_factor, 22),
                                                tar_id=id_render,
                                                angular_dist_method='dist',
                                                scene_center=world_center)
        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)


        image_list = [train_rgb_files[i].split('/')[-1] for i in nearest_pose_ids]
        datadir = self.scene_dir[idx]
        
        images = load_rgbs(image_list, os.path.join(datadir, 'images'),
                           self.depth_H, self.depth_W, is_png=True)
        
        depths, masks = load_colmap(image_list, datadir,
                                    self.depth_H, self.depth_W,)
        depths = torch.from_numpy(depths)
        masks = torch.from_numpy(masks)
        #print(images.shape, depths.shape, masks.shape)
        return {
            'images': images,
            'depths': depths,
            'masks': masks
        }        


if __name__ == "__main__":
    from dotmap import DotMap
    args = DotMap()
    args.datadir = "/media/hdd1/Datasets/DTU_NerfingMVS"
    args.list_prefix = 'pixelnerf'
    args.factor = 1
    args.num_source_views = 10
    args.depth_H = 256
    args.depth_W = 320

    dataset = DTUMVSNeRFDataset(args, mode='train')
    for data in dataset:
        print('hello')
        print(data["images"].shape, data["depths"].shape, data["masks"].shape)
        raise NotImplementedError
