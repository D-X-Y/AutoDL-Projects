# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from os import path as osp
from copy import deepcopy as copy
from tqdm import tqdm
import warnings, time, random, numpy as np

from pts_utils import generate_label_map
from xvision import denormalize_points
from xvision import identity2affine, solve2theta, affine2image
from .dataset_utils import pil_loader
from .landmark_utils import PointMeta2V
from .augmentation_utils import CutOut
import torch
import torch.utils.data as data


class LandmarkDataset(data.Dataset):
    def __init__(
        self,
        transform,
        sigma,
        downsample,
        heatmap_type,
        shape,
        use_gray,
        mean_file,
        data_indicator,
        cache_images=None,
    ):

        self.transform = transform
        self.sigma = sigma
        self.downsample = downsample
        self.heatmap_type = heatmap_type
        self.dataset_name = data_indicator
        self.shape = shape  # [H,W]
        self.use_gray = use_gray
        assert transform is not None, "transform : {:}".format(transform)
        self.mean_file = mean_file
        if mean_file is None:
            self.mean_data = None
            warnings.warn("LandmarkDataset initialized with mean_data = None")
        else:
            assert osp.isfile(mean_file), "{:} is not a file.".format(mean_file)
            self.mean_data = torch.load(mean_file)
        self.reset()
        self.cutout = None
        self.cache_images = cache_images
        print("The general dataset initialization done : {:}".format(self))
        warnings.simplefilter("once")

    def __repr__(self):
        return "{name}(point-num={NUM_PTS}, shape={shape}, sigma={sigma}, heatmap_type={heatmap_type}, length={length}, cutout={cutout}, dataset={dataset_name}, mean={mean_file})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def set_cutout(self, length):
        if length is not None and length >= 1:
            self.cutout = CutOut(int(length))
        else:
            self.cutout = None

    def reset(self, num_pts=-1, boxid="default", only_pts=False):
        self.NUM_PTS = num_pts
        if only_pts:
            return
        self.length = 0
        self.datas = []
        self.labels = []
        self.NormDistances = []
        self.BOXID = boxid
        if self.mean_data is None:
            self.mean_face = None
        else:
            self.mean_face = torch.Tensor(self.mean_data[boxid].copy().T)
            assert (self.mean_face >= -1).all() and (
                self.mean_face <= 1
            ).all(), "mean-{:}-face : {:}".format(boxid, self.mean_face)
        # assert self.dataset_name is not None, 'The dataset name is None'

    def __len__(self):
        assert len(self.datas) == self.length, "The length is not correct : {}".format(
            self.length
        )
        return self.length

    def append(self, data, label, distance):
        assert osp.isfile(data), "The image path is not a file : {:}".format(data)
        self.datas.append(data)
        self.labels.append(label)
        self.NormDistances.append(distance)
        self.length = self.length + 1

    def load_list(self, file_lists, num_pts, boxindicator, normalizeL, reset):
        if reset:
            self.reset(num_pts, boxindicator)
        else:
            assert (
                self.NUM_PTS == num_pts and self.BOXID == boxindicator
            ), "The number of point is inconsistance : {:} vs {:}".format(
                self.NUM_PTS, num_pts
            )
        if isinstance(file_lists, str):
            file_lists = [file_lists]
        samples = []
        for idx, file_path in enumerate(file_lists):
            print(
                ":::: load list {:}/{:} : {:}".format(idx, len(file_lists), file_path)
            )
            xdata = torch.load(file_path)
            if isinstance(xdata, list):
                data = xdata  # image or video dataset list
            elif isinstance(xdata, dict):
                data = xdata["datas"]  # multi-view dataset list
            else:
                raise ValueError("Invalid Type Error : {:}".format(type(xdata)))
            samples = samples + data
        # samples is a dict, where the key is the image-path and the value is the annotation
        # each annotation is a dict, contains 'points' (3,num_pts), and various box
        print("GeneralDataset-V2 : {:} samples".format(len(samples)))

        # for index, annotation in enumerate(samples):
        for index in tqdm(range(len(samples))):
            annotation = samples[index]
            image_path = annotation["current_frame"]
            points, box = (
                annotation["points"],
                annotation["box-{:}".format(boxindicator)],
            )
            label = PointMeta2V(
                self.NUM_PTS, points, box, image_path, self.dataset_name
            )
            if normalizeL is None:
                normDistance = None
            else:
                normDistance = annotation["normalizeL-{:}".format(normalizeL)]
            self.append(image_path, label, normDistance)

        assert (
            len(self.datas) == self.length
        ), "The length and the data is not right {} vs {}".format(
            self.length, len(self.datas)
        )
        assert (
            len(self.labels) == self.length
        ), "The length and the labels is not right {} vs {}".format(
            self.length, len(self.labels)
        )
        assert (
            len(self.NormDistances) == self.length
        ), "The length and the NormDistances is not right {} vs {}".format(
            self.length, len(self.NormDistance)
        )
        print(
            "Load data done for LandmarkDataset, which has {:} images.".format(
                self.length
            )
        )

    def __getitem__(self, index):
        assert index >= 0 and index < self.length, "Invalid index : {:}".format(index)
        if self.cache_images is not None and self.datas[index] in self.cache_images:
            image = self.cache_images[self.datas[index]].clone()
        else:
            image = pil_loader(self.datas[index], self.use_gray)
        target = self.labels[index].copy()
        return self._process_(image, target, index)

    def _process_(self, image, target, index):

        # transform the image and points
        image, target, theta = self.transform(image, target)
        (C, H, W), (height, width) = image.size(), self.shape

        # obtain the visiable indicator vector
        if target.is_none():
            nopoints = True
        else:
            nopoints = False
        if index == -1:
            __path = None
        else:
            __path = self.datas[index]
        if isinstance(theta, list) or isinstance(theta, tuple):
            affineImage, heatmaps, mask, norm_trans_points, THETA, transpose_theta = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for _theta in theta:
                (
                    _affineImage,
                    _heatmaps,
                    _mask,
                    _norm_trans_points,
                    _theta,
                    _transpose_theta,
                ) = self.__process_affine(
                    image, target, _theta, nopoints, "P[{:}]@{:}".format(index, __path)
                )
                affineImage.append(_affineImage)
                heatmaps.append(_heatmaps)
                mask.append(_mask)
                norm_trans_points.append(_norm_trans_points)
                THETA.append(_theta)
                transpose_theta.append(_transpose_theta)
            affineImage, heatmaps, mask, norm_trans_points, THETA, transpose_theta = (
                torch.stack(affineImage),
                torch.stack(heatmaps),
                torch.stack(mask),
                torch.stack(norm_trans_points),
                torch.stack(THETA),
                torch.stack(transpose_theta),
            )
        else:
            (
                affineImage,
                heatmaps,
                mask,
                norm_trans_points,
                THETA,
                transpose_theta,
            ) = self.__process_affine(
                image, target, theta, nopoints, "S[{:}]@{:}".format(index, __path)
            )

        torch_index = torch.IntTensor([index])
        torch_nopoints = torch.ByteTensor([nopoints])
        torch_shape = torch.IntTensor([H, W])

        return (
            affineImage,
            heatmaps,
            mask,
            norm_trans_points,
            THETA,
            transpose_theta,
            torch_index,
            torch_nopoints,
            torch_shape,
        )

    def __process_affine(self, image, target, theta, nopoints, aux_info=None):
        image, target, theta = image.clone(), target.copy(), theta.clone()
        (C, H, W), (height, width) = image.size(), self.shape
        if nopoints:  # do not have label
            norm_trans_points = torch.zeros((3, self.NUM_PTS))
            heatmaps = torch.zeros(
                (self.NUM_PTS + 1, height // self.downsample, width // self.downsample)
            )
            mask = torch.ones((self.NUM_PTS + 1, 1, 1), dtype=torch.uint8)
            transpose_theta = identity2affine(False)
        else:
            norm_trans_points = apply_affine2point(target.get_points(), theta, (H, W))
            norm_trans_points = apply_boundary(norm_trans_points)
            real_trans_points = norm_trans_points.clone()
            real_trans_points[:2, :] = denormalize_points(
                self.shape, real_trans_points[:2, :]
            )
            heatmaps, mask = generate_label_map(
                real_trans_points.numpy(),
                height // self.downsample,
                width // self.downsample,
                self.sigma,
                self.downsample,
                nopoints,
                self.heatmap_type,
            )  # H*W*C
            heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1))).type(
                torch.FloatTensor
            )
            mask = torch.from_numpy(mask.transpose((2, 0, 1))).type(torch.ByteTensor)
            if self.mean_face is None:
                # warnings.warn('In LandmarkDataset use identity2affine for transpose_theta because self.mean_face is None.')
                transpose_theta = identity2affine(False)
            else:
                if torch.sum(norm_trans_points[2, :] == 1) < 3:
                    warnings.warn(
                        "In LandmarkDataset after transformation, no visiable point, using identity instead. Aux: {:}".format(
                            aux_info
                        )
                    )
                    transpose_theta = identity2affine(False)
                else:
                    transpose_theta = solve2theta(
                        norm_trans_points, self.mean_face.clone()
                    )

        affineImage = affine2image(image, theta, self.shape)
        if self.cutout is not None:
            affineImage = self.cutout(affineImage)

        return affineImage, heatmaps, mask, norm_trans_points, theta, transpose_theta
