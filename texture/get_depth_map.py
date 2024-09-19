import sys, os
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import cv2
from matplotlib import cm
import matplotlib.colors
import csv
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
from matplotlib import colors
import matplotlib.ticker as ticker
def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def read_depth(filename):
    depth = read_pfm(filename)[0]
    return np.array(depth, dtype=np.float32)

#获取GT与pred_depth之间的error图(%)
'''GT_depth_filename = "/home/ubuntu/home/wck/texture/GT_error/depth_map_0028_GT.pfm"
mask_filename = "/home/ubuntu/home/wck/texture/GT_error/depth_visual_0028.png"
depth_filename = "/home/ubuntu/home/wck/texture/GT_error/00000028.pfm"
depth = read_depth(depth_filename)
gt_depth = read_depth(GT_depth_filename)
im = np.array(Image.open(mask_filename))
gt_depth = cv2.resize(gt_depth,(400,296))
im = cv2.resize(im,(400,296))
mask = im > 10
depth = depth*mask
depth_max = np.max(np.abs(depth-gt_depth))
error = np.divide((depth-gt_depth),510, out=np.zeros_like(depth))
error = np.abs(error)

error_max = np.max(error)

error = cv2.resize(error,(1152,864))
plt.figure(figsize=(8,6))
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
viridis = cm.get_cmap('hot', 256)
newcolors = viridis(np.linspace(0, 1, 256))
white = np.array([255 / 255, 255 / 255, 255 / 255, 1])
newcolors[:1, :] = white
norm1 = mpl.colors.Normalize(vmin=0, vmax=1)
cmap = ListedColormap(newcolors)
plt.imshow(error,cmap=cmap)
plt.axis('off')
# plt.savefig('./Geo_15_28_error.png')
plt.savefig('./MVSNet_15_28_error.png')
# plt.colorbar(mpl.cm.ScalarMappable(norm=norm1,cmap=cmap))
# plt.savefig('./bar_001.png')
plt.show()'''

#输出深度图
# mask_filename = "/home/ubuntu/home/wck/texture/GT_error/depth_visual_0028.png"
depth_filename = "/home/ubuntu/home/wck/texture/TAT_prob/00000141_P.pfm"
depth = read_depth(depth_filename)
# im = np.array(Image.open(mask_filename))
# im = cv2.resize(im,(1600,1152))
# mask = im > 10
# depth = depth*mask
plt.figure(figsize=(30,17))
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)

# depth = cv2.resize(depth,(1920,1088))
# viridis = cm.get_cmap('hot', 256)
# newcolors = viridis(np.linspace(0, 1, 256))
# white = np.array([255 / 255, 255 / 255, 255 / 255, 1])
# newcolors[:1, :] = white
# norm1 = mpl.colors.Normalize(vmin=0, vmax=1)
# cmap = ListedColormap(newcolors)
plt.imshow(depth,cmap="viridis_r")
plt.axis('off')
# plt.savefig('./Geo_15_28_error.png')
# plt.savefig('./scan1_1.png')
# plt.colorbar(mpl.cm.ScalarMappable(norm=norm1,cmap=cmap))
#plt.savefig('./family_141_P.png')
plt.show()
