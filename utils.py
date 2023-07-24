import os, gzip, torch
import pandas as pd
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


# coordinates of the data points
data_points = 7744
df_xz = pd.read_csv('./dataset/XZ_coor.csv', header=None)
data_xz = np.asarray(df_xz)
coor_x = data_xz[0, :data_points] - np.min(data_xz[0])
coor_z = data_xz[1, :data_points] - np.min(data_xz[1])

# print structure of betwork
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# save iamges of every epoch
def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    # fig = np.squeeze(merge(images, size))

    nums = np.random.randint(0, size*size, size)
    fig, axs = plt.subplots(1, size, figsize=(size, 8), sharey=True)
    for index, num in enumerate(nums):
        color = images[index, :, :, :].ravel()
        ax = axs[index].scatter(coor_x, coor_z, s = 8, c= 255 * color, marker = 's', alpha = 0.8)
    plt.colorbar(ax)
    return plt.savefig(path)

# merge images
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

# generate animation of generated images
def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.v2.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    fig = plt.figure()
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
