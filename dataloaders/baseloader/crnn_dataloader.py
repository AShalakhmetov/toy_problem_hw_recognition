from __future__ import print_function, division

import torch.utils.data as data
import numpy as np
import torch
import cv2
import random

import os
import os.path

from utils import utils

from PIL import Image
from torchvision import datasets, transforms

LETTERS_ALPHA_ = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T',
                  'X', 'Y']

IMG_EXTENSIONS_ = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CRNNImageDatasetFolder(data.Dataset):
    """An extended data loader, designed for C-RNN arhitecture. All samples are located in same folder, filenames stand for labels: ::

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, loader=default_loader, transform=transforms.ToTensor(),target_transform=transforms.ToTensor()):
        self.extensions = IMG_EXTENSIONS_
        self.loader = loader
        self.samples = []

        self.transform = transform
        self.target_transform = target_transform
        self.timestepsize = None
        self.chars = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, target_len) where target is label and target_len is a length of the label.
        """
        assert self.samples is not None
        path, target = self.samples[index]
        sample = self.loader(path)
        target_len = [len(target)]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            npa = np.asarray(target)
            target = torch.from_numpy(npa)

        return sample, target, target_len

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format('Defined in child class!')
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __makedir__(self, path):
        try:
            # Create target Directory
            os.mkdir(path)
            print("Directory ", path, " Created ")
        except FileExistsError:
            print("Directory ", path, " already exists")


    def make_dataset(self, pth, extensions=IMG_EXTENSIONS_):
        """
        This method creates dataset from folder with images. Filename of each image file is used as a label for sample.
        Each character in label encoded as an index of total character list (indices start from 1, thus 0-index is reserved for blank character).

        :param pth:         Path to the dataset
        :param chars:       Total list of characters
        :param extensions:  List of allowed extensions

        :return:            List of tuples of path to image file and its encoded label
        """
        assert pth is not None and self.chars is not None
        images = []
        dir = os.path.expanduser(pth)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)

            if has_file_allowed_extension(target, extensions):
                fname = os.path.splitext(target)[0]
                # item = (d, text_to_labels(fname, chars))
                item = (d, utils.encode(fname, self.chars))
                images.append(item)
        return images


    def __set_samples__(self, samples):
        self.samples = samples

    def __settimesteps__(self, timestepsize):
        self.timestepsize = timestepsize

    def __set_chars__(self, chars):
        self.chars = chars

    def __set_target_transforms__(self, transform):
        self.target_transform = transform

    def __gettimesteps__(self):
        assert self.timestepsize is not None
        return self.timestepsize

    def __getclassesnum__(self):
        pass

    def __get_chars__(self):
        assert self.chars is not None
        return self.chars

    def get_random_sample(self):
        rand = random.randrange(0, self.__len__(), 1)
        return self.__getitem__(rand)