import torch
import numpy as np
import cv2
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def reshape_tensor(x, shape=(5, 75)):
    # return x.reshape(x.size[0] // 3, 3, x.size[1], x.size[2])
    return x.reshape(shape)


data = ImageFolder(root='./abc/', transform=torchvision.transforms.ToTensor())
loader = DataLoader(data)

for batch_idx, (data, target) in enumerate(loader):
    print("Initial img")
    print(data, data.shape)

    p = data.permute(0, 3, 2, 1)
    print("Permuted image")
    print(p, p.shape)
    p = p.contiguous()

    # a = data.view(1, -1, )
    a = p.view(1, -1, )

    print("Viewed image")
    print(a, 2 * '\n', a.shape, 3 * '\n')


    x = a.view(1, 5, -1)
    print('View = 5 image')
    print(x, 2 * '\n', x.shape)






# img = cv2.imread('123.png')
# img = img/255
#
# print("Initial img")
# print(img, 2*'\n', img.shape,  3*'\n')
#
# t = torch.Tensor(img)
# p = t.permute(1,2,0)
#
# print("Permuted image")
# print(p, p.shape)
# p = p.contiguous()
# a = p.view(-1,)
#
# print("Viewed image")
# print(a, 2*'\n', a.shape, 3*'\n')
# #
# # # x = reshape_tensor(a)
# x = a.view(5, -1)
#
# print("View = 5 image")
# print(x, 2*'\n', x.shape)