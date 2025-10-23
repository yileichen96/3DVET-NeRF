import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import math

sys.path.append("../")
from .data_utils import random_crop, random_flip, get_nearest_pose_ids, parse_camera_pre
from .llff_data_utils import load_llff_data, batch_parse_llff_poses


class LLFFDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        base_dir = os.path.join(args.dataset_dir, "TrainingSet/real_iconic_noface/")
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        # scenes = os.listdir(base_dir)
        with open(os.path.join(base_dir, 'scene_list.txt'), 'r') as file:
            scenes = file.read().splitlines()
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(base_dir, scene)
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(scene_path, load_imgs=False, factor=4)
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            if mode == "train":
                i_train = np.array(np.arange(int(poses.shape[0])))
                i_render = i_train
            else:
                i_test = np.arange(poses.shape[0])[:: self.args.llffhold]
                i_train = np.array(
                    [
                        j
                        for j in np.arange(int(poses.shape[0]))
                        if (j not in i_test and j not in i_test)
                    ]
                )
                i_render = i_test

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
            self.render_train_set_ids.extend([i] * num_render)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        if self.mode == "train":
            id_render = train_rgb_files.index(rgb_file)
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=3)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            min(self.num_source_views * subsample_factor, 20),
            tar_id=id_render,
            angular_dist_method="dist",
        )

        # if self.mode == "train":
        #     id_render = train_rgb_files.index(rgb_file)
        # else:
        #     id_render = -1
        # num_select = 20
        #
        # nearest_pose_ids = get_nearest_pose_ids(
        #     render_pose,
        #     train_poses,
        #     num_select,
        #     tar_id=id_render,
        #     angular_dist_method="dist",
        # )
        nearest_pose_ids = np.random.choice(
            nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False
        )

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        if self.mode == "train":
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(
                rgb, camera, src_rgbs, src_cameras, (crop_h, crop_w)
            )

        if self.mode == "train" and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])

        # computing proj_mats
        W, H, intrinsics, c2w_target = parse_camera_pre(camera)
        w2c_target = np.linalg.inv(c2w_target)
        intrinsic = intrinsics[:3, :3].copy()
        feat_scale_factor = np.stack([W / math.ceil(W / 4),
                                      H / math.ceil(H / 4)]).reshape(2, 1)
        intrinsic[:2] = intrinsic[:2] / feat_scale_factor
        tar_proj_inv = np.eye(4)
        tar_proj_inv[:3, :4] = intrinsic @ w2c_target[:3, :4]
        tar_proj_inv = np.linalg.inv(tar_proj_inv)

        proj_mats = []
        for id in range(len(nearest_pose_ids)):
            # build proj mat from source views to target view
            W, H, intrinsics, c2w = parse_camera_pre(src_cameras[id])
            w2c = np.linalg.inv(c2w)
            intrinsic = intrinsics[:3, :3].copy()
            feat_scale_factor = np.stack([W / math.ceil(W / 4),
                                          H / math.ceil(H / 4)]).reshape(2, 1)
            intrinsic[:2] = intrinsic[:2] / feat_scale_factor
            proj_mat_l = np.eye(4)
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            proj_mat = proj_mat_l @ tar_proj_inv
            proj_mats.append(proj_mat[:3])

        proj_mats = np.stack(proj_mats)

        return {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_file,
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
            "proj_mats": torch.from_numpy(proj_mats).float(),
        }
