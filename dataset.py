from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from scipy.io import loadmat, savemat
from imgaug import augmenters as iaa
import imgaug as ia
#from augmentator import ImgAugTransform
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class CaliforniaDataset(torch.utils.data.Dataset):
    """Some Information about CaliforniaDataset"""
    def __init__(self, img_size, data_size, mat_file):
        super(CaliforniaDataset, self).__init__()

        self.dataset_dict = loadmat(mat_file)
        x = self.dataset_dict["t1_L8_clipped"]
        y = self.dataset_dict["logt2_clipped"]
        c_x = x.shape[-1]
        c_y = y.shape[-1]
        x_chs = slice(0, c_x, 1)
        y_chs = slice(c_x, c_x + c_y, 1)
        print(x.shape)
        print(y.shape)
        print(type(x))
        w, h = x.shape[0:2]
        input = np.append(x, y, axis= -1)
        gt = self.dataset_dict["ROI"]
        gt = np.expand_dims(gt, axis = -1)

        self.transforms = transforms.Compose([
            #transforms.ToPILImage(),
            #ImgAugTransform
        ])

        self.dataset_x = []
        self.dataset_y = []
        self.labels = []
        for i in range(data_size//10):
            data = input
            for j in range(10):
                random_x = np.random.randint(low = 0, high = w - img_size - 1)
                random_y = np.random.randint(low = 0, high = h - img_size - 1)
                img = data[random_x : random_x + img_size, random_y : random_y + img_size, :]
                label = gt[random_x : random_x + img_size, random_y : random_y + img_size, :]
                self.dataset_x.append(img[:, :, x_chs])
                self.dataset_y.append(img[:, :, y_chs])
                self.labels.append(label)
        self.dataset_x = self.swapaxes_(torch.tensor(self.dataset_x))
        self.dataset_y = self.swapaxes_(torch.tensor(self.dataset_y))
        self.labels = torch.tensor(self.labels)
        print("x shape: ", self.dataset_x.shape)
        print("y_shape: ", self.dataset_y.shape)
        print("gt_shape: ", self.labels.shape)
        #shape


    def __getitem__(self, index):
        return ((self.dataset_x[index, :, :, :], self.dataset_y[index, :, :, :]), self.labels[index, : ,:, :])

    def __len__(self):
        return len(self.dataset_x)

    def swapaxes_(self, dataset):
        dataset = torch.swapaxes(dataset, 1, -1)
        dataset = torch.swapaxes(dataset, -2, -1)
        return dataset
   