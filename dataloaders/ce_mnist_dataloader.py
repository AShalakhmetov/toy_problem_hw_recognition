from __future__ import print_function, division

import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from dataloaders.baseloader.crnn_dataloader import CRNNImageDatasetFolder

NUMBERS_ = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

class CEMNISTDataloader(CRNNImageDatasetFolder):
    def __init__(self, sample_size=5):
        super().__init__(CEMNISTDataloader, self)
        self.sample_size_ = sample_size
        self.__set_chars__(chars=NUMBERS_)
        self.root = self.__gendataset__(self.sample_size_)
        samples = self.make_dataset(self.root)
        self.__set_samples__(samples)
        self.__settimesteps__(self.sample_size_)

        self.sample_w, self.sample_h = 28 * self.sample_size_, 28  # Predefined values


    def __gendataset__(self, sample_size, dataset_size=10000):
        mnist = datasets.MNIST(
            './mnist',
            train=True,
            download=True,
            transform=None,
            target_transform=None)

        root_dir = './e_mnist'
        self.__makedir__(root_dir)

        if len(os.listdir(root_dir)) > 0:
            print('Dataset folder is not empty')
            return root_dir

        for i in range(dataset_size):
            np.random.seed(1234)
            img = None
            label = list()

            for j in range(sample_size):
                im, l = self.__getrandomsample__(mnist)
                if j == 0:
                    img = im
                else:
                    assert img is not None
                    img = np.concatenate((img, im), axis=1)

                label.append(l)

            ret_img = Image.fromarray(img, mode='L')
            path = root_dir + '/' + ''.join(label) + '.png'
            ret_img.save(path)

            if i % 100 == 0:
                print(">>> Generation of CE MNIST Dataset {}% complete      ".format(int(i/100)), end='\r', flush=True)

        print("CE MNIST Dataset created")
        return root_dir

    def __getrandomsample__(self, dataset):
        ri = random.randrange(0, dataset.__len__(), 1)
        im, l = dataset.__getitem__(ri)
        if isinstance(l, torch.Tensor):
            l = l.numpy()
        return np.array(im), str(l)

    def __plus_minus_proba__(self, a):
        plus = random.uniform(0, a)
        minus = random.uniform(0, a)
        return plus, minus

    def __getclassesnum__(self):
        """
        Returns number of character classes. NOTE THAT 'blank' CHARACTER IS EXCLUDED AND MUST BE TAKEN INTO ACCOUNT.
        :return: Number of classes ('blank' excluded)
        """
        return len(NUMBERS_)

    def __getsamplesize__(self):
        """
        Returns sample size. Each sample in MNIST dataset equals 28x28 pixels.
        :return: Width and height of each sample
        """
        return self.sample_w, self.sample_h