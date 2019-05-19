from __future__ import print_function, division

import torch.utils.data as data
import numpy as np
import cv2
import random

import os
import os.path

from PIL import Image
from torchvision import datasets, transforms

LETTERS_ALPHA_ = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T',
                  'X', 'Y']
NUMBERS_ = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
IMG_EXTENSIONS_ = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

TYPE_PLATE_ = 'PLATE_RECO'
TYPE_MNIST_E_ = 'EXTENDED_MNIST_RECO'
TYPE_IAM_ = 'I_AM_RECO'


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def text_to_labels(text, letters):
    return list(map(lambda x: letters.index(x), text))


def make_dataset(pth, extensions, chars=LETTERS_ALPHA_):
    images = []
    dir = os.path.expanduser(pth)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)

        if has_file_allowed_extension(target, extensions):
            fname = os.path.splitext(target)[0]
            item = (d, text_to_labels(fname, chars))
            images.append(item)
            #         item = (target, text_to_labels(fname, chars))
            #         images.append(item)

    return images


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


class ImageDatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

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
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, type=TYPE_MNIST_E_,
                 loader=default_loader, transform=transforms.ToTensor(),
                 target_transform=None):
        #                  target_transform=transforms.ToTensor()):
        #         classes, class_to_idx = text_to_labels(root)

        self.root, self.extensions, self.chars = self.__genparams__(type)

        if root is not None and type != TYPE_MNIST_E_:
            self.root = root

        self.loader = loader

        samples = make_dataset(self.root, self.extensions, self.chars)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(IMG_EXTENSIONS_)))

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target,

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
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

    def __gendataset__(self, type=TYPE_MNIST_E_, dataset_size=10000, sample_size=5):
        mnist = datasets.MNIST(
            './mnist',
            train=True,
            download=True,
            transform=None)

        root_dir = './e_mnist'
        self.__makedir__(root_dir)

        for i in range(dataset_size):
            np.random.seed(1234)

            im1, l1 = self.__getrandomsample__(mnist)
            im2, l2 = self.__getrandomsample__(mnist)
            im3, l3 = self.__getrandomsample__(mnist)
            im4, l4 = self.__getrandomsample__(mnist)
            im5, l5 = self.__getrandomsample__(mnist)

            pp1, pm1 = self.__plus_minus_proba__(0.2)
            pp2, pm2 = self.__plus_minus_proba__(0.2)
            pp3, pm3 = self.__plus_minus_proba__(0.2)
            pp4, pm4 = self.__plus_minus_proba__(0.2)
            pp5, pm5 = self.__plus_minus_proba__(0.2)

            v1 = im1
            v2 = im2
            v3 = im3
            v4 = im4
            v5 = im5

            # v1 = im1[im1.shape[0] * pp1:im1.shape[0] * -pm1, :]
            # v2 = im2[pp2:-pm2, :]
            # v3 = im3[pp3:-pm3, :]
            # v4 = im4[pp4:-pm4, :]
            # v5 = im5[pp5:-pm5, :]

            # v1 = im1[:, int(round(im1.shape[0]*pp1)):]
            # v2 = im2[:, :int(round(im2.shape[0]* -pm2))]
            # v3 = im3[:, int(round(im3.shape[0]*pp3)):]
            # v4 = im4[:, :int(round(im4.shape[0]*-pm4))]
            # v5 = im5[:, :int(round(im5.shape[0]*-pm5))]

            # print(v1.shape)
            # print(v2.shape)
            # print(v3.shape)
            # print(v4.shape)


            img = np.concatenate((v1, v2, v3, v4, v5), axis=1)
            full = np.zeros((28, 140))

            # offset_plus, offset_minus  = self.__plus_minus_proba__(0.1)
            # if offset_plus >= offset_minus:
            #     full[offset_plus:]

            # full[:, :img.shape[1]] = img
            full = img
            label = [l1, l2, l3, l4, l5]

            ret_img = Image.fromarray(full, mode='L')
            path = root_dir + '/' + ''.join(label) + '.png'
            ret_img.save(path)

        return root_dir

    def __genparams__(self, type):
        if type == TYPE_PLATE_:
            return self.root, IMG_EXTENSIONS_, LETTERS_ALPHA_
        elif type == TYPE_MNIST_E_:
            root_dir = self.__gendataset__()
            return root_dir, IMG_EXTENSIONS_, NUMBERS_
        elif type == TYPE_IAM_:
            raise (RuntimeError("IAM dataset is not implemented yet."))
        else:
            raise (RuntimeError("No such type."))

    def __getrandomsample__(self, dataset):
        # if not isinstance(dataset, datasets):
        #     raise (RuntimeError("Passed object is not a dataset."))

        # ri = np.random.randint(dataset.__len__(), size=1)
        ri = random.randrange(0, dataset.__len__(), 1)
        im, l = dataset.__getitem__(ri)
        return np.array(im), str(l)

    def __plus_minus_proba__(self, a):
        plus = random.uniform(0, a)
        minus = random.uniform(0, a)
        return plus, minus


idf = ImageDatasetFolder(root=None, type=TYPE_MNIST_E_)