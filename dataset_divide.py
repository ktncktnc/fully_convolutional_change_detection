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
import random
#from augmentator import ImgAugTransform
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class CaliforniaDataset(torch.utils.data.Dataset):
    """Some Information about CaliforniaDataset"""
    def __init__(self, img_size, data_size, imgs_1, imgs_2, gt, transform = None):
        super(CaliforniaDataset, self).__init__()
        self.imgs_1 = torch.from_numpy(imgs_1)
        self.imgs_2 = torch.from_numpy(imgs_2)
        self.gt = torch.from_numpy(gt)
        self.transform = transform

        p_count = self.imgs_1.shape[0]
        w, h = self.imgs_1.shape[2:]
        self.img_size = img_size

        n_pix = img_size[0]*img_size[1]*data_size
        true_pix = 0

        self.coords = []
        for _ in range(data_size):
            
            patch = np.random.randint(low = 0, high = p_count - 1)
            #print(patch)
            random_x = np.random.randint(low = 0, high = w - img_size[0] - 1)
            random_y = np.random.randint(low = 0, high = h - img_size[1] - 1)
            true_pix += np.sum(self.gt[patch, random_x:random_x + img_size[0], random_y:random_y + img_size[1]].numpy())
            self.coords.append([patch, random_x, random_y])

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
            'label': self.gt[coord[0], coord[1]:coord[1] + self.img_size[0], coord[2]:coord[2] + self.img_size[1]],
            'I1': self.imgs_1[coord[0], :, coord[1]:coord[1] + self.img_size[0], coord[2]:coord[2] + self.img_size[1]],
            'I2': self.imgs_2[coord[0], :, coord[1]:coord[1] + self.img_size[0], coord[2]:coord[2] + self.img_size[1]]
            
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


    
def dataset_split(mat_file):

    dataset_dict = loadmat(mat_file)
    imgs_1 = dataset_dict["t1_L8_clipped"]
    imgs_2 = dataset_dict["logt2_clipped"]
    gt = dataset_dict["ROI"]

    imgs_1 = imgs_1.transpose(2, 0, 1)
    imgs_2 = imgs_2.transpose(2, 0, 1)

    rand_list = []
    proba = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
    for i in range(28):
        num = random.randint(0, len(proba) - 1)
        rand_list.append(proba[num])
        proba.remove(rand_list[-1])

    train_imgs1 = []
    test_imgs1 = []
    val_imgs1 = []
    train_imgs2 = []
    test_imgs2 = []
    val_imgs2 = []
    train_gt = []
    test_gt = []
    val_gt = []

    for i in range(7):
        for j in range(4):
            if rand_list[4*i + j] == 0:
                train_imgs1.append(imgs_1[:, i*500:(i+1)*500, j*500:(j + 1)*500])
                train_imgs2.append(imgs_2[:, i*500:(i+1)*500, j*500:(j + 1)*500])
                train_gt.append(gt[i*500:(i+1)*500, j*500:(j + 1)*500])
            elif rand_list[4*i + j] == 1:
                val_imgs1.append(imgs_1[:, i*500:(i+1)*500, j*500:(j + 1)*500])
                val_imgs2.append(imgs_2[:, i*500:(i+1)*500, j*500:(j + 1)*500])
                val_gt.append(gt[i*500:(i+1)*500, j*500:(j + 1)*500])
            else:
                test_imgs1.append(imgs_1[:, i*500:(i+1)*500, j*500:(j + 1)*500])
                test_imgs2.append(imgs_2[:, i*500:(i+1)*500, j*500:(j + 1)*500])
                test_gt.append(gt[i*500:(i+1)*500, j*500:(j + 1)*500])
    
    train_imgs1 = np.asarray(train_imgs1)
    train_imgs2 = np.asarray(train_imgs2)
    train_gt = np.asarray(train_gt)

    val_imgs1 = np.asarray(val_imgs1)
    val_imgs2 = np.asarray(val_imgs2)
    val_gt = np.asarray(val_gt)

    test_imgs1 = np.asarray(test_imgs1)
    test_imgs2 = np.asarray(test_imgs2)
    test_gt = np.asarray(test_gt)
                
    return (train_imgs1, train_imgs2, train_gt), (val_imgs1, val_imgs2, val_gt), (test_imgs1, test_imgs2, test_gt)