from __future__ import print_function, division

import torch.utils.data as data
from torchvision import get_image_backend, datasets, transforms

import numpy as np
import cv2
import random

import os
import os.path

from PIL import Image

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
	return list(map(lambda x: letters.index(x) + 1, text))


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
				#                  target_transform=None):
				target_transform=transforms.ToTensor()):
				#classes, class_to_idx = text_to_labels(root)

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
		target_len = [len(target)]

		if self.transform is not None:
			sample = self.transform(sample)
		if self.transform is not None:
			npa = np.asarray(target)
			target = torch.from_numpy(npa)

		return sample, target, target_len

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
		"""
		Args:
			type (str) : specified type of dataset
			dataset_size (int) : number of samples in dataset
			sample_size (int) : number of variables in sample, e.g. number of letters/digits
		Returns:
			root_folder (str) : path to generated dataset
		"""
		if type == TYPE_MNIST_E_:
			return self.__generate_e_mnist_dataset__(dataset_size, sample_size)
		elif type == TYPE_PLATE_:
			return None

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

	def __generate_e_mnist_dataset__(self, dataset_size, sample_size):
		mnist = datasets.MNIST(
			'./mnist',
			train=True,
			download=True,
			transform=None)

		root_dir = './e_mnist'
		self.__makedir__(root_dir)

		for i in range(dataset_size):
			np.random.seed(1234)
			full = np.zeros((28, 28 * sample_size))
			img = []
			whole_label = []

			for j in range(sample_size):
				im, l = self.__getrandomsample__(mnist)
				arr = np.ravel(im)
				img.append(arr)
				whole_label.append(l)
            
			i = np.concatenate(img, axis=1)
			full = i
			label = whole_label

			ret_img = Image.fromarray(full, mode='L')
			path = root_dir + '/' + ''.join(label) + '.png'
			ret_img.save(path)

		return root_dir