import os
from os import listdir
from os.path import join
from torch.utils.data import Dataset
from newTransform import compose, normalize, toTensor, randomCrop, randomHorizontalFlip, randomVerticalFlip, \
    randomRotation, Resize_train, Resize_test, randomRoll

import scipy.io as scio
import numpy as np
import torch

from kornia.filters import get_gaussian_kernel2d

import kornia


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".mat"])


def calculate_valid_crop_size(upscale_factor, patchSize):
    return upscale_factor / patchSize


def train_hr_transform(crop_size):
    return compose([
        randomCrop(crop_size),
        randomHorizontalFlip(),
        randomVerticalFlip(),
        randomRoll(),
        randomRotation('90'),
        randomRotation('-90'),
        toTensor()
    ])


def train_lr_transform(upscale_factor, interpolation):
    return compose([
        Resize_train(upscale_factor, interpolation)
    ])


class TrainsetFromFolder(Dataset):
    def __init__(self, dataset, dataset_dir, upscale_factor, interpolation='Bicubic', patchSize=64, crop_num=16):
        super(TrainsetFromFolder, self).__init__()

        train_dir1 = join(dataset_dir, "hsi")
        train_dir2 = join(dataset_dir, "rgb")
        #train_dir3 = join(dataset_dir, "lr_hsi")

        self.image_filenames1 = [join(train_dir1, x) for x in listdir(train_dir1) if is_image_file(x)]
        self.image_filenames2 = [join(train_dir2, x) for x in listdir(train_dir2) if is_image_file(x)]
        self.image_filenames1 = sorted(self.image_filenames1)
        self.image_filenames2 = sorted(self.image_filenames2)
        self.xs = []
        self.ys = []

        for img in self.image_filenames1:
            self.xs.append(load_img(img))

        for img in self.image_filenames2:
            self.ys.append(load_img1(img))


        self.hr_transform = train_hr_transform(patchSize)
        self.lr_transform = train_lr_transform(upscale_factor, interpolation)

        self.lens = 10000

    def __getitem__(self, index):
        ind = index % 12
        hsi = self.xs[ind]
        rgb = self.ys[ind]
        cat = np.concatenate((hsi, rgb), axis=2)

        crop = self.hr_transform(cat)

        crop_hsi = crop[:128, :, :]
        crop_hr_rgb = crop[128:131, :, :]

        crop_lr_hsi = my_gaussian_blur2d(crop_hsi.unsqueeze(0), (3,3), (0.5,0.5)).squeeze(0)
        crop_lr_hsi = self.lr_transform(crop_lr_hsi)

        return crop_lr_hsi, crop_hr_rgb, crop_hsi

    def __len__(self):
        return self.lens


def test_lr_hsi_transform(upscale_factor, interpolation):
    return compose([
        Resize_test(upscale_factor, interpolation),
        toTensor()
    ])


def test_hr_rgb_transform():
    return compose([
        toTensor()
    ])


def mrop(scale, width, height):
    W = width // scale
    H = height // scale

    return int(W * scale), int(H * scale)


class ValsetFromFolder(Dataset):
    def __init__(self, dataset, dataset_dir, upscale_factor, interpolation):
        super(ValsetFromFolder, self).__init__()
        test_dir1 = join(dataset_dir, "hsi")
        test_dir2 = join(dataset_dir, "rgb")
        self.image_filenames1 = [join(test_dir1, x) for x in listdir(test_dir1) if is_image_file(x)]
        self.image_filenames2 = [join(test_dir2, x) for x in listdir(test_dir1) if is_image_file(x)]

        self.image_filenames1 = sorted(self.image_filenames1)
        self.image_filenames2 = sorted(self.image_filenames2)

        self.hsi = []
        self.rgb = []

        for img in self.image_filenames1:
            self.hsi.append(load_img(img))

        for img in self.image_filenames2:
            self.rgb.append(load_img1(img))


        self.lr_hsi_transform = test_lr_hsi_transform(upscale_factor, interpolation)
        self.hr_rgb_transform = test_hr_rgb_transform()
        self.scale = upscale_factor

    def __getitem__(self, index):

        hsi = self.hsi[index].permute(2,0,1)
        rgb = self.rgb[index].permute(2,0,1)

        W, H = mrop(self.scale, hsi.shape[1], hsi.shape[2])

        hsi = hsi[:, :W, :H]
        rgb = rgb[:, :W, :H]


        lr_hsi = self.lr_hsi_transform(hsi)

        hr_rgb = self.hr_rgb_transform(rgb)

        hsi = self.hr_rgb_transform(hsi)

        return lr_hsi, hr_rgb, hsi, self.image_filenames1[index]

    def __len__(self):
        return len(self.image_filenames1)


def load_img(filepath):
    x = scio.loadmat(filepath)
    #x = x['msi'].astype(np.float32)
    x = x['hsi'].astype(np.float32)
    x = torch.tensor(x).float()
    return x


def load_img1(filepath):
    x = scio.loadmat(filepath)
    x = x['msi'].astype(np.float32)
    #x = x['RGB'].astype(np.float32)
    x = torch.tensor(x).float()
    return x


def load_img2(filepath):
    x = scio.loadmat(filepath)
    x = x['blur'].astype(np.float32)
    x = torch.tensor(x).float()
    return x


def my_gaussian_blur2d(input, kernel_size, sigma, border_type = 'reflect'):

    kernel = torch.unsqueeze(get_gaussian_kernel2d(kernel_size, sigma, force_even=True), dim=0)
    # print(kernel)

    return kornia.filters.filter2d(input, kernel, border_type)
