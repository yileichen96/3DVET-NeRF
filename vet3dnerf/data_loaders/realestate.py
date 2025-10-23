import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import cv2
import math

from .data_utils import parse_camera_pre


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.intrinsics = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def unnormalize_intrinsics(intrinsics, h, w):
    intrinsics[0] *= w
    intrinsics[1] *= h
    return intrinsics


def parse_pose_file(file):
    f = open(file, "r")
    cam_params = {}
    for i, line in enumerate(f):
        if i == 0:
            continue
        entry = [float(x) for x in line.split()]
        id = int(entry[0])
        cam_params[id] = Camera(entry)
    return cam_params


# only for training
class RealEstateDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.folder_path = os.path.join(args.dataset_dir, "TrainingSet/RealEstate10K-subset/")
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.target_h, self.target_w = 450, 800
        assert mode in ["train", "test"]
        # self.scene_path_list = glob.glob(os.path.join(self.folder_path, mode, "frames", "*"))

        with open(os.path.join(self.folder_path, 'scene_list.txt'), 'r') as file:
            scene_names = file.read().splitlines()
        self.scene_path_list = [os.path.join(self.folder_path, mode, "frames", scene_name) for scene_name in scene_names]
        
        all_rgb_files = []
        all_timestamps = []
        for i, scene_path in enumerate(self.scene_path_list):
            rgb_files = [os.path.join(scene_path, f) for f in sorted(os.listdir(scene_path))]
            if len(rgb_files) < 10:
                print("omitting {}, too few images".format(os.path.basename(scene_path)))
                continue
            timestamps = [int(os.path.basename(rgb_file).split(".")[0]) for rgb_file in rgb_files]
            sorted_ids = np.argsort(timestamps)
            all_rgb_files.append(np.array(rgb_files)[sorted_ids])
            all_timestamps.append(np.array(timestamps)[sorted_ids])

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files, dtype=object)[index]
        self.all_timestamps = np.array(all_timestamps, dtype=object)[index]

    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):
        rgb_files = self.all_rgb_files[idx]
        timestamps = self.all_timestamps[idx]

        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)
        window_size = 32
        shift = np.random.randint(low=-1, high=2)
        id_render = np.random.randint(low=4, high=num_frames - 4 - 1)

        right_bound = min(id_render + window_size + shift, num_frames - 1)
        left_bound = max(0, right_bound - 2 * window_size)
        candidate_ids = np.arange(left_bound, right_bound)
        # remove the query frame itself with high probability
        if np.random.choice([0, 1], p=[0.01, 0.99]):
            candidate_ids = candidate_ids[candidate_ids != id_render]

        id_feat = np.random.choice(
            candidate_ids, size=min(self.num_source_views, len(candidate_ids)), replace=False
        )
        # num_select = 20
        # id_feat = np.random.choice(
        #     candidate_ids, size=min(num_select, len(candidate_ids)), replace=False
        # )

        rgb_file = rgb_files[id_render]
        rgb = imageio.imread(rgb_files[id_render])
        # resize the image to target size
        rgb = cv2.resize(rgb, dsize=(self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32) / 255.0

        camera_file = os.path.dirname(rgb_file).replace("frames", "cameras") + ".txt"
        cam_params = parse_pose_file(camera_file)
        cam_param = cam_params[timestamps[id_render]]

        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (
                list(img_size),
                unnormalize_intrinsics(
                    cam_param.intrinsics, self.target_h, self.target_w
                ).flatten(),
                cam_param.c2w_mat.flatten(),
            )
        ).astype(np.float32)

        # get depth range
        depth_range = torch.tensor([1.0, 100.0])

        src_rgbs = []
        src_cameras = []
        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id])
            # resize the image to target size
            src_rgb = cv2.resize(
                src_rgb, dsize=(self.target_w, self.target_h), interpolation=cv2.INTER_AREA
            )
            src_rgb = src_rgb.astype(np.float32) / 255.0
            src_rgbs.append(src_rgb)

            img_size = src_rgb.shape[:2]
            cam_param = cam_params[timestamps[id]]
            src_camera = np.concatenate(
                (
                    list(img_size),
                    unnormalize_intrinsics(
                        cam_param.intrinsics, self.target_h, self.target_w
                    ).flatten(),
                    cam_param.c2w_mat.flatten(),
                )
            ).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs)
        src_cameras = np.stack(src_cameras)

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
        for id in range(len(id_feat)):
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
            "rgb": torch.from_numpy(rgb),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_files[id_render],
            "src_rgbs": torch.from_numpy(src_rgbs),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
            "proj_mats": torch.from_numpy(proj_mats).float(),
        }
