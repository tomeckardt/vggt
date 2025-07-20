# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os.path as osp
import os
import logging

import random
import h5py
import numpy as np


from data.dataset_util import *
from data.base_dataset import BaseDataset

def cam_to_opencv(cam_T: np.ndarray):
    y_z_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    cam_T = y_z_swap.dot(cam_T)
    # revert x-axis
    cam_T[:3, 0] *= -1

    # opengl camera to opencv camera
    R = cam_T[:3, :3]
    T = cam_T[:3, 3]
    R[:, 1] *= -1
    R[:, 2] *= -1

    cam_T_world = np.eye(4)
    cam_T_world[:3, :3] = R
    x, z, y = cam_T[:3, 3]
    cam_T_world[:3, 3] = [x, y, z]

    
    return cam_T_world


class Renderings3DFrontDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        RENDERINGS_DIR: str | None = None,
        min_num_images: int = 6,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the Renderings3DFrontDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            RENDERINGS_DIR (str): Directory path to CO3D data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If RENDERINGS_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = False # common_conf.allow_duplicate_img

        if RENDERINGS_DIR is None:
            raise ValueError("RENDERINGS_DIR must be specified.")
        
        with open(osp.join(RENDERINGS_DIR, f"selected_seqs_{split}.json")) as f:
            scene_data = json.load(f)

        if self.debug:
            category = ["0a9c667d-033d-448c-b17c-dc55e6d3c386"]

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence = [] # set any invalid sequence names here


        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"RENDERINGS_DIR is {RENDERINGS_DIR}")

        self.RENDERINGS_DIR = RENDERINGS_DIR

        total_frame_num = 0
        
        for scene_name, rooms in scene_data.items():
            for room_name, view_ids in rooms.items():
                if len(view_ids) < min_num_images or room_name in self.invalid_sequence:
                    continue
                total_frame_num += len(view_ids)
                self.data_store[scene_name + "/" + room_name] = view_ids

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Co3D Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Co3D Data dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int | None = None,
        img_per_seq: int | None = None,
        seq_name: str | None = None,
        ids: np.ndarray | list | None = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        
        assert seq_index is not None
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]
        assert seq_name is not None

        metadata = self.data_store[seq_name]
        print(seq_name, metadata)

        assert img_per_seq is not None
        if ids is None:
            print(len(metadata), metadata)
            ids = np.random.choice(
                metadata, img_per_seq, replace=self.allow_duplicate_img
            )
            print(ids)

        assert ids is not None

        target_image_shape = self.get_target_shape(aspect_ratio)

        intri_opencv = np.load(os.path.join(self.RENDERINGS_DIR, seq_name.split("/")[0], "cam_K.npy"))

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for i in ids:
            image_path = os.path.join(self.RENDERINGS_DIR, *seq_name.split("/"), str(i) + ".hdf5")
            with h5py.File(image_path) as f:
                image = np.array(f["colors"])
                if self.load_depth:
                    depth_map = np.array(f["depth"])
                    depth_map = threshold_depth_map(
                        depth_map, min_percentile=-1, max_percentile=98
                    )
                else:
                    depth_map = None
                extri_opencv = np.array(f["cam_Ts"])
                extri_opencv = cam_to_opencv(extri_opencv)
                extri_opencv = np.linalg.inv(extri_opencv)
                extri_opencv = extri_opencv[:3]
            original_size = np.array(image.shape[:2])

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        set_name = "3dfront_renderings"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch

if __name__ == "__main__":
    from hydra import initialize, compose
    from types import SimpleNamespace
    with initialize(config_path="../../config"):
        cfg = compose(config_name="default")
        common_config = SimpleNamespace(**cfg["data"]["train"]["common_config"])
        common_config.augs = SimpleNamespace(**common_config.augs)
        common_config.augs.color_jitter = SimpleNamespace(**common_config.augs.color_jitter) \
            if common_config.augs.color_jitter is not None else None
    
    dataset = Renderings3DFrontDataset(
        common_config, 
        split="test", 
        RENDERINGS_DIR="/usr/prakt/s0018/3d_scene_editing/renderings"
    )
    batch = dataset.get_data(img_per_seq=6)
    print(batch.keys())
    
    import rerun as rr
    rr.init("VGGT Co3D", spawn=False)
    rr.connect_grpc()
    for i, (colors, points, extrinsics, intrinsics) \
        in enumerate(zip(batch["images"], batch["world_points"], batch["extrinsics"], batch["intrinsics"]), 1):
        print(extrinsics)
        extrinsics = np.linalg.inv(np.concatenate((extrinsics, [[0, 0, 0, 1]])))
        rr.log(f"pts{i}", rr.Points3D(points.reshape(-1, 3), colors=colors.reshape(-1, 3)))
        rr.log(
            f'cam{i}', 
            rr.Pinhole(
                resolution=colors.shape[:2],
                focal_length=float(intrinsics[0, 0]),
                camera_xyz=rr.ViewCoordinates.RDF, 
                image_plane_distance=0.2,
            )
        )

        rr.log(
            f"cam{i}",
            rr.Transform3D(translation=extrinsics[:3,3], mat3x3=extrinsics[:3,:3]),
        )
        rr.log(f"cam{i}", rr.Image(colors))