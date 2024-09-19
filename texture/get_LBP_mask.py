import cv2 as cv
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import re

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)


def calute_basic_lbp(image_array, i, j):
    num = 0
    if image_array[i - 1, j - 1] != image_array[i, j]:
        num += 1
    if image_array[i - 1, j] != image_array[i, j]:
        num += 1
    if image_array[i - 1, j + 1] != image_array[i, j]:
        num += 1
    if image_array[i, j - 1] != image_array[i, j]:
        num += 1
    if image_array[i, j + 1] != image_array[i, j]:
        num += 1
    if image_array[i + 1, j - 1] != image_array[i, j]:
        num += 1
    if image_array[i + 1, j] != image_array[i, j]:
        num += 1
    if image_array[i + 1, j + 1] != image_array[i, j]:
        num += 1
    return num


def lbp_basic(image_array):
    basic_array = np.zeros(image_array.shape, np.uint8)
    width = image_array.shape[0]
    height = image_array.shape[1]
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            num = calute_basic_lbp(image_array, i, j)
            basic_array[i, j] = num

    mask0 = basic_array == 0
    mask1 = basic_array == 1
    mask2 = basic_array == 2
    mask3 = basic_array == 3
    mask4 = basic_array == 4
    mask5 = basic_array == 5
    mask6 = basic_array == 6
    mask7 = basic_array == 7
    mask8 = basic_array == 8
    mask = {"0": mask0.astype(float),
            "1": mask1.astype(float),
            "2": mask2.astype(float),
            "3": mask3.astype(float),
            "4": mask4.astype(float),
            "5": mask5.astype(float),
            "6": mask6.astype(float),
            "7": mask7.astype(float),
            "8": mask8.astype(float),
            }
    return mask


def read_img(filename):
    img = Image.open(filename)
    gray_image = img.convert('L')
    gray_image = np.array(gray_image, dtype=np.float32)
    return gray_image


def read_img_1(filename):
    img_ = Image.open(filename)
    np_img = np.array(img_, dtype=np.float32)
    return np_img


def get_mask(ref_lbp_mask, mask):
    lbp_mask_0 = ref_lbp_mask["0"]
    lbp_mask_1 = ref_lbp_mask["1"]
    lbp_mask_2 = ref_lbp_mask["2"]
    lbp_mask_3 = ref_lbp_mask["3"]
    lbp_mask_4 = ref_lbp_mask["4"]
    lbp_mask_5 = ref_lbp_mask["5"]
    lbp_mask_6 = ref_lbp_mask["6"]
    lbp_mask_7 = ref_lbp_mask["7"]
    lbp_mask_8 = ref_lbp_mask["8"]

    mask = mask > 10
    mask = mask.astype(int)

    final_mask = ((lbp_mask_0 + lbp_mask_1 + lbp_mask_2 + lbp_mask_3 + lbp_mask_4 + lbp_mask_5) != 0)

    # final_mask = np.logical_and(final_mask, mask)
    ratio = np.sum(final_mask == True) / (1200 * 1600)
    valid_mask = np.where(final_mask, 0, 255)
    # valid_mask = np.where(final_mask, 0, 1)
    # valid_mask = valid_mask * mask

    return valid_mask


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


def fill_in_fast(depth_filter_, max_depth=1.0005, custom_kernel=DIAMOND_KERNEL_5,
                 blur_type='gaussian'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """
    depth_map = depth_filter_
    # mask depth>1
    mask_depth1 = (depth_map != 0)
    # Invert
    valid_pixels = (depth_map > 0.0)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    save_depth_map(depth_map,"inverse_pic.png")

    # Dilate
    depth_map = cv.dilate(depth_map, custom_kernel)

    save_depth_map(depth_map, "Dilate_DIAMOND_KERNEL_5.png")

    # Hole closing
    depth_map = cv.morphologyEx(depth_map, cv.MORPH_CLOSE, FULL_KERNEL_5)

    save_depth_map(depth_map, "MORPH_CLOSE_FULL_KERNEL_5.png")

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map == 0)
    dilated = cv.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    save_depth_map(depth_map,"Dilate_FULL_KERNEL_7.png")

    empty_pixels = depth_map < 0.1
    dilated = cv.dilate(depth_map, FULL_KERNEL_31)
    depth_map[empty_pixels] = dilated[empty_pixels]

    save_depth_map(depth_map,"Dilate_FULL_KERNEL_31.png")

    # Median blur
    depth_map = cv.medianBlur(depth_map, 5)

    save_depth_map(depth_map,"mediaBlur.png")

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.0)
        blurred = cv.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]
        save_depth_map(depth_map,"Gaussian_blur.png")

    # Invert
    valid_pixels = (depth_map > 0.0)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    return depth_map


def read_depth(filename):
    depth = read_pfm(filename)[0]
    return np.array(depth, dtype=np.float32)


def save_depth_map(pic, name):
    pic = pic * (933.8 - 425) + 425
    plt.figure(dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(pic)
    plt.axis("off")
    plt.savefig(name)
    plt.show()


if __name__ == '__main__':
    img_filename = "/home/ubuntu/home/wck/data/mvs_training/dtu/Rectified_raw/scan1/rect_003_0_r5000.png"
    mask_filename = "/home/ubuntu/home/wck/data/mvs_training/dtu/Depths_raw/Depths/scan1/depth_visual_0000.png"
    gray_img = read_img(img_filename)
    mask_img = read_img(mask_filename)
    lbp_uniform = local_binary_pattern(gray_img.astype(int), 8, 1, method="uniform")
    ref_lbp_mask = lbp_basic(lbp_uniform)
    final_mask = get_mask(ref_lbp_mask, mask_img)

    # final_mask = cv.resize(final_mask, (1600, 1152), interpolation=cv.INTER_NEAREST)

    # depth_filename = "/home/ubuntu/home/wck/GeoMVSNet_test_1/outputs/dtu/geomvsnet_raw/scan1/depth_est/00000000.pfm"
    #
    # depth = read_depth(depth_filename)
    #
    # pic = (depth - 425.0) / (933.8 - 425.0)
    #
    # pic = pic * final_mask

    # pic_filter = fill_in_fast(pic.astype("float32"))
    plt.figure(dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(final_mask,cmap="gray")
    plt.axis("off")
    plt.savefig("lbp_mask3.png")
    plt.show()

    # save_depth_map(pic_filter,"pic_filter.png")
