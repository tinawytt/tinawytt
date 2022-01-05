#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

from medpy.io import load
import os
import numpy as np
import torch

import random
from copy import deepcopy
from scipy.ndimage import map_coordinates, fourier_gaussian
from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_magnitude
from scipy.ndimage.morphology import grey_dilation
from skimage.transform import resize
from scipy.ndimage.measurements import label as lb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import SimpleITK as sitk
from utilities.file_and_folder_operations import subfiles

def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit
    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).
    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer


def preprocess_data(root_dir, y_shape=64, z_shape=64):
    # image_dir = os.path.join(root_dir, 'imagesTr')
    image_dir = root_dir
    label_dir = os.path.join(root_dir, 'labelsTr')
    output_dir = os.path.join(root_dir, 'preprocessed')
    classes = 4

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    class_stats = defaultdict(int)
    total = 0
    nii_files=[]
    subfiles(image_dir,nii_files, suffix=".nii.gz", join=False)

    # for i in range(0, len(nii_files)):
    #     if nii_files[i].startswith("._"):
    #         nii_files[i] = nii_files[i][2:]
    # print("--------")

    for f in nii_files:
        # print(os.path.join(image_dir, f))
        # image, _ = load(os.path.join(image_dir, f))
        # label, _ = load(os.path.join(label_dir, f.replace('_0000', '')))
        image,metadata=load(f)
        # plt.imshow(image, cmap = cm.Greys_r)
        print(1,image)
        label=metadata.get_sitkimage()
        npImage = sitk.GetArrayFromImage(label)
        print(2,npImage)
        z = int(label.GetSize()[2]/2)
        slice = sitk.GetArrayFromImage(label)[z,:,:]
        plt.figure(figsize=(5,5))
        plt.imshow(slice, 'gray')
        plt.show()
        for i in range(classes):
            class_stats[i] += np.sum(label == i)
            total += np.sum(label == i)

        # normalize images
        image = (image - image.min())/(image.max()-image.min())

        image = pad_nd_image(image, (image.shape[0], y_shape, z_shape), "constant", kwargs={'constant_values': image.min()})
        label = pad_nd_image(label, (image.shape[0], y_shape, z_shape), "constant", kwargs={'constant_values': label.min()})

        result = np.stack((image, label))

        np.save(os.path.join(output_dir, f.split('.')[0]+'.npy'), result)
        print(f)

    print(total)
    for i in range(classes):
        print(class_stats[i], class_stats[i]/total)


def preprocess_single_file(image_file):
    image, image_header = load(image_file)
    image = (image - image.min()) / (image.max() - image.min())

    data = np.expand_dims(image, 1)

    return torch.from_numpy(data), image_header


def postprocess_single_image(image):
    # desired shape is [b w h]
    result_converted = image[::, 0, ::, ::]
    result_mapped = [i * 255 for i in result_converted]

    return result_mapped
