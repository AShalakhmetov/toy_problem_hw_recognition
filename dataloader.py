from __future__ import print_function, division

import torch.utils.data as data
import numpy as np
import cv2

import os
import os.path

from PIL import Image


LETTERS_ALPHA_ = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']
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
  

def make_dataset(dir, extensions, chars=LETTERS_ALPHA_):
    images = []
    dir = os.path.expanduser(train_path)
    for target in sorted(os.listdir(dir)):
      d = os.path.join(dir, target)
  
      if has_file_allowed_extension(target, extensions):
        fname = os.path.splitext(target)[0]
        item = (d, text_to_labels(fname, chars))
        images.append(item)
        item = (target, text_to_labels(fname, chars))
        images.append(item)

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

    def __init__(self, root, type=TYPE_PLATE_,
                 loader=default_loader, extensions=IMG_EXTENSIONS, chars=LETTERS, transform=None, target_transform=None):
#         classes, class_to_idx = text_to_labels(root)



        samples = make_dataset(root, extensions, chars)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                            "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

#         self.classes = classes
#         self.class_to_idx = class_to_idx
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
        
        # img = cv2.imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (128, 64))
        # img = img.astype(np.float32)
        # img /= 255
        # sample = img

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # sample = torch.Tensor(sample)
        # target = torch.Tensor(target)
        
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
