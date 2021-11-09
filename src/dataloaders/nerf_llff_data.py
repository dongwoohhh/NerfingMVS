# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys

sys.path.append('../..')

#from .data_utils import rectify_inplane_rotation, random_crop, random_flip, get_nearest_pose_ids
#from .llff_data_utils import load_llff_data, batch_parse_llff_poses
from .data_utils import rectify_inplane_rotation, random_crop, random_flip, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses

import numpy as np
import torch
from torch.utils.data import Dataset
import glob

from utils.io_utils import *



class NerfLLFFDataset(Dataset):
    def __init__(self, args, mode,
                 # scenes=('chair', 'drum', 'lego', 'hotdog', 'materials', 'mic', 'ship'),
                 scenes=(), **kwargs):
        self.args = args
        self.datadir = args.datadir

        self.num_source_views = args.num_source_views
        self.rectify_inplane_rotation = False

        self.depth_H = args.depth_H
        self.depth_W = args.depth_W

        self.mode = mode

        self.datadir = os.path.join(args.datadir, 'data/nerf_llff_data/')

        scenedir = [x for x in glob.glob(os.path.join(self.datadir, "*")) if os.path.isdir(x)]

        scenes = ('fern', 'flower', 'horns', 'room', 'orchids', 'leaves', 'fortress', 'trex')
        all_scenes = []
        for scene in scenedir:
            if scene.split('/')[-1] in scenes:
                all_scenes.append((scene.split('/')[-1], scene))
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
                i_train = np.array(np.arange(int(poses.shape[0])))
                i_render = i_train
                """
                raise NotImplementedError
                i_test = np.arange(poses.shape[0])[::args.llffhold]
                i_train = np.array([j for j in np.arange(int(poses.shape[0])) if
                                    (j not in i_test and j not in i_test)])
                i_render = i_test
                """

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
        return len(self.scene_dir)
    

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
        subsample_factor = 2
        num_select = self.num_source_views * subsample_factor

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                #min(self.num_source_views*subsample_factor, 22),
                                                num_select,
                                                tar_id=id_render,
                                                angular_dist_method='dist',
                                                scene_center=world_center)
        
        #nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)

        nearest_pose_ids = np.random.choice(nearest_pose_ids, num_select, replace=False)
        support_pose_ids = nearest_pose_ids[:self.num_source_views]
        query_pose_ids = nearest_pose_ids[self.num_source_views:]

        support_images, support_depths, support_masks = self._load_sample(idx, train_rgb_files, support_pose_ids)
        query_images, query_depths, query_masks = self._load_sample(idx, train_rgb_files, query_pose_ids)

        #print(query_images.shape, query_depths.shape, query_masks.shape)
        #print(support_images.shape, support_depths.shape, support_masks.shape)
        
        #print(images.shape, depths.shape, masks.shape)

        return {
            "support_images": support_images,
            "support_depths": support_depths,
            "support_masks": support_masks,
            "query_images": query_images,
            "query_depths": query_depths,
            "query_masks": query_masks,
            #"scene_id":
        }
    def _load_sample(self, idx, rgb_files, pose_ids):
        image_list = [rgb_files[i].split('/')[-1] for i in pose_ids]
        datadir = self.scene_dir[idx]
        images = load_rgbs(image_list, os.path.join(datadir, 'images'),
                           self.depth_H, self.depth_W, is_png=False)
        depths, masks = load_colmap(image_list, datadir,
                                    self.depth_H, self.depth_W)
        
        depths = torch.from_numpy(depths)
        masks = torch.from_numpy(masks)
        #print(datadir, masks.sum())
        return images, depths, masks

if __name__ == "__main__":
    from dotmap import DotMap
    args = DotMap()
    args.datadir = "/media/hdd1/Datasets/ibrnet_NerfingMVS"
    #args.list_prefix = 'pixelnerf'
    args.factor = 2
    args.num_source_views = 4
    args.depth_H = 400#800
    args.depth_W = 400#800

    dataset = NerfLLFFDataset(args, mode='train')
    for data in dataset:
        print('hello')
        print(data["support_images"].shape, data["support_depths"].shape, data["support_masks"].shape)
        raise NotImplementedError
