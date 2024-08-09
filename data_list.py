#from __future__ import print_function, division

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
from data.utils import noisify, noisify_instance, noisify_instance_v2

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, noisy=True, noise_type=None, noise_rate=0, random_state=0, num_class=0, labels=None, \
                  transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.noisy = noisy
        self.transform = transform
        self.target_transform = target_transform
        num_class = num_class
        idx_each_class_noisy = [[] for i in range(num_class)]
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        if noisy == True:
            self.train_data = []
            self.train_labels = []
            for (path, label) in self.imgs:
                self.train_labels.append([label])
            
            if noise_type != 'instance':
                self.train_labels = np.asarray(self.train_labels)
                self.train_noisy_labels, self.actual_noise_rate = noisify(dataset='visda', nb_classes=num_class, train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state)
                self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                _train_labels=[i[0] for i in self.train_labels]
                self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)
            else:
                for (path, label) in self.imgs:
                    self.train_data.append(self.transform(self.loader(path)))
                self.train_data = torch.tensor(np.array([np.array(item) for item in self.train_data])) #convert list to tensor 
                self.train_noisy_labels, self.actual_noise_rate = noisify_instance(self.train_data, self.train_labels, num_class, noise_rate)
                print('over all noise rate is ', self.actual_noise_rate)
                for i in range(len(self.train_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(num_class)]
                self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                print(f'The noisy data ratio in each class is {self.noise_prior}')
                _train_labels=[i[0] for i in self.train_labels]
                self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)
            
            self.imgs = np.asarray(self.imgs)
            self.imgs[:,1] = np.asarray(self.train_noisy_labels)
            self.imgs = self.imgs.tolist()
        
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, int(target), index

    def __len__(self):
        return len(self.imgs)

class ImageValueList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=rgb_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.values = [1.0] * len(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values):
        self.values = values

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

