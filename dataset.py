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
    def __init__(self, img_size, data_size, mat_file, transform = None):
        super(CaliforniaDataset, self).__init__()

        dataset_dict = loadmat(mat_file)
        self.imgs_1 = dataset_dict["t1_L8_clipped"]
        self.imgs_2 = dataset_dict["logt2_clipped"]
        self.gt = dataset_dict["ROI"]
        self.imgs_1 = self.swapaxes_(self.imgs_1)
        self.imgs_2 = self.swapaxes_(self.imgs_2)
        self.gt = torch.from_numpy(self.gt)

        self.transform = transform

        #self.imgs_1 shape: 

        #print(x.shape)
        #print(y.shape)
        #print(type(x))


        w, h = self.imgs_1.shape[1:]
        self.img_size = img_size

        n_pix = img_size[0]*img_size[1]*data_size
        true_pix = 0

        self.coords = []
        for _ in range(data_size):
            random_x = np.random.randint(low = 0, high = w - img_size[0] - 1)
            random_y = np.random.randint(low = 0, high = h - img_size[1] - 1)
            true_pix += np.sum(self.gt[random_x:random_x + img_size[0], random_y:random_y + img_size[1]].numpy())
            self.coords.append([random_x, random_y])

        print("imgs_1 shape: ", self.imgs_1.shape)
        print("imgs_2 shape: ", self.imgs_2.shape)
        print("gt shape: ", self.gt.shape)
        print("true_pix: ", true_pix, "npix: ", n_pix)
        print(self.coords)
        self.weights = [ 10 * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]
        #shape


    def __getitem__(self, index):
        coord = self.coords[index]
        sample = {
            'label': self.gt[coord[0]:coord[0] + self.img_size[0], coord[1]:coord[1] + self.img_size[1]],
            'I1': self.imgs_1[:, coord[0]:coord[0] + self.img_size[0], coord[1]:coord[1] + self.img_size[1]],
            'I2': self.imgs_2[:, coord[0]:coord[0] + self.img_size[0], coord[1]:coord[1] + self.img_size[1]]
            
        }
        
        if self.transform is not None:
            sample = self.transform(sample)
        #print('index = ', index)
        #print('I1 shape' , sample['I1'].shape)
        #print('I2 shape' ,sample['I2'].shape)
        #print('label shape' ,sample['label'].shape)
        return sample

    def __len__(self):
        return len(self.coords)

    def swapaxes_(self, I):
        return torch.from_numpy(I.transpose(2, 0, 1))