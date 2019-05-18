import os
import json
from os.path import join
from collections import Counter
import matplotlib as plt

def get_counter(dirpath, tag):
    dirname = os.path.basename(dirpath)
    dirpath = join(dirpath, 'ann')
    letters = ''
    lens = []
    for filename in os.listdir(dirpath):
        json_filepath = join(dirpath, filename)
        ann = json.load(open(json_filepath, 'r'))
        tags = ann['tags']
        if tag in tags:
            description = ann['description']
            lens.append(len(description))
            letters += description
    print('Max plate length in "%s":' % dirname, max(Counter(lens).keys()))
    return Counter(letters)


c_val = get_counter('/data/anpr_ocr__train', 'val')
c_train = get_counter('/data/anpr_ocr__train', 'train')
letters_train = set(c_train.keys())
letters_val = set(c_val.keys())
if letters_train == letters_val:
    print('Letters in train and val do match')
else:
    raise Exception()
# print(len(letters_train), len(letters_val), len(letters_val | letters_train))
letters = sorted(list(letters_train))
print('Letters:', ' '.join(letters))


def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))


def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            return False
    return True

import numpy as np
from collections import Counter
import os
from os.path import join
import json
import random
import cv2

class CustomDataLoader():
    def __init__(self,dirpath,
                 tag,
                 img_w, img_h,
                 batch_size,
                 downsample_factor,
                 max_text_len=8):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor

        img_dirpath = join(dirpath, 'img')
        ann_dirpath = join(dirpath, 'ann')
        self.samples = []
        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            if ext in ['.png', '.jpg']:
                img_filepath = join(img_dirpath, filename)
                json_filepath = join(ann_dirpath, name + '.json')
                ann = json.load(open(json_filepath, 'r'))
                description = ann['description']
                tags = ann['tags']
                if tag not in tags:
                    continue
                if is_valid_str(description):
                    self.samples.append([img_filepath, description])

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0

    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            img = cv2.imread(img_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img /= 255
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = img
            self.texts.append(text)

    def get_output_size(self):
        return len(letters) + 1

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            # if K.image_data_format() == 'channels_first':
            #     X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            # else:
            #     X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                # if K.image_data_format() == 'channels_first':
                #     img = np.expand_dims(img, 0)
                # else:
                #     img = np.expand_dims(img, -1)
                img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = text_to_labels(text)
                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                # 'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)


cdl = CustomDataLoader('/data/anpr_ocr__train', 'val', 128, 64, 8, 4)
cdl.build_data()

for inp, out in cdl.next_batch():
    print('Text generator output (data which will be fed into the neutral network):')
    print('1) the_input (image)')
    # if K.image_data_format() == 'channels_first':
    #     img = inp['the_input'][0, 0, :, :]
    # else:
    #     img = inp['the_input'][0, :, :, 0]
    img = inp['the_input'][0, :, :, 0]

    plt.imshow(img.T, cmap='gray')
    plt.show()
    print('2) the_labels (plate number): %s is encoded as %s' %
          (labels_to_text(inp['the_labels'][0]), list(map(int, inp['the_labels'][0]))))
    print('3) input_length (width of image that is fed to the loss function): %d == %d / 4 - 2' %
          (inp['input_length'][0], cdl.img_w))
    print('4) label_length (length of plate number): %d' % inp['label_length'][0])
    break