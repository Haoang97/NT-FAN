from PIL import Image
import torch.utils.data as data
import os
import os.path
import numpy as np
#from typing import Any, Callable, Optional, Tuple
from .utils import download_url, check_integrity, noisify, noisify_instance

class SVHN(data.Dataset):
    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, 
                 noise_type=None, noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.dataset = 'svhn'
        idx_each_class_noisy = [[] for i in range(10)]

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        if self.split == 'train':
            loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

            self.train_data = loaded_mat['X']
            self.train_labels = loaded_mat['y'].astype(np.int64).squeeze()

            np.place(self.train_labels, self.train_labels == 10, 0)
            self.train_data = np.transpose(self.train_data, (3, 2, 0, 1))

            if noise_type != None:
                if noise_type != 'instance':
                    self.train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                    self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state)
                    self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
                    _train_labels = [i[0] for i in self.train_labels]
                    self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)
                else:
                    self.train_noisy_labels, self.actual_noise_rate = noisify_instance(self.train_data, self.train_labels,noise_rate=noise_rate)
                    print('over all noise rate is ', self.actual_noise_rate)
                    for i in range(len(self.train_labels)):
                        idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                    self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')
                    self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)

        else:
            loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

            self.test_data = loaded_mat['X']
            self.test_labels = loaded_mat['y'].astype(np.int64).squeeze()

            np.place(self.test_labels, self.test_labels == 10, 0)
            self.test_data = np.transpose(self.test_data, (3, 2, 0, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'train':
            if self.noise_type is not None:
                img, target = self.train_data[index], int(self.train_noisy_labels[index])
            else:
                img, target = self.train_data[index], int(self.train_labels[index])
        else:
            img, target = self.test_data[index], int(self.test_labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


    def __len__(self):
        if self.split == 'train':
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)