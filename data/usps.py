from PIL import Image
import os
import numpy as np
import torch
from .utils import download_url
#from .vision import VisionDataset
from .utils import noisify, noisify_instance
from typing import cast

class USPS(torch.utils.data.Dataset):
    """`USPS <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps>`_ Dataset.
    The data-format is : [label [index:value ]*256 \\n] * num_lines, where ``label`` lies in ``[1, 10]``.
    The value for each pixel lies in ``[-1, 1]``. Here we transform the ``label`` into ``[0, 9]``
    and make pixel values in ``[0, 255]``.

    Args:
        root (string): Root directory of dataset to store``USPS`` data files.
        train (bool, optional): If True, creates dataset from ``usps.bz2``,
            otherwise from ``usps.t.bz2``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    split_list = {
        'train': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
            "usps.bz2", 'ec16c51db3855ca6c91edd34d0e9b197'
        ],
        'test': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
            "usps.t.bz2", '8ea070ee2aca1ac39742fdd1ef5ed118'
        ],
    }

    def __init__(self, root, train = True, transform = None, target_transform = None, download = False,
                 noise_type=None, noise_rate=0.2, random_state=0):
        super(USPS, self).__init__()
        self.dataset = 'usps'
        self.noise_type = noise_type
        self.split = 'train' if train else 'test'
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        idx_each_class_noisy = [[] for i in range(10)]
        url, filename, checksum = self.split_list[self.split]
        full_path = os.path.join(self.root, filename)

        if download and not os.path.exists(full_path):
            download_url(url, self.root, filename, md5=checksum)

        import bz2
        if self.split == 'train':
            with bz2.open(full_path) as fp:
                raw_data = [line.decode().split() for line in fp.readlines()]
                tmp_list = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
                imgs = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16))
                imgs = ((cast(np.ndarray, imgs) + 1) / 2 * 255).astype(dtype=np.uint8)
                targets = [int(d[0]) - 1 for d in raw_data]
            self.data = imgs
            self.targets = targets
            if noise_type != None:
                if noise_type != 'instance':
                    self.targets = np.asarray([[self.targets[i]] for i in range(len(self.targets))])
                    self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, train_labels=self.targets, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state)
                    self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
                    _targets=[i[0] for i in self.targets]
                    self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_targets)
                else:
                    self.train_noisy_labels, self.actual_noise_rate = noisify_instance(self.data, self.targets,noise_rate=noise_rate)
                    print('over all noise rate is ', self.actual_noise_rate)
                    for i in range(len(self.targets)):
                        idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                    self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.targets)
        else:
            with bz2.open(full_path) as fp:
                raw_data = [line.decode().split() for line in fp.readlines()]
                tmp_list = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
                imgs = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16))
                imgs = ((cast(np.ndarray, imgs) + 1) / 2 * 255).astype(dtype=np.uint8)
                targets = [int(d[0]) - 1 for d in raw_data]
            self.data = imgs
            self.targets = targets
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'train':
            if self.noise_type is not None:
                img, target = self.data[index], int(self.train_noisy_labels[index])
            else:
                img, target = self.data[index], int(self.targets[index])
        else:
            img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


    def __len__(self):
        return len(self.data)
