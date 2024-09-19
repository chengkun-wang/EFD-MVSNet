import cv2
import numpy as np
import csv
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

def fill_in_fast(depth_filter_, max_depth=1.00001, custom_kernel=DIAMOND_KERNEL_5,
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
    depth_map = depth_filter_.copy()
    depth_map = depth_map.astype(np.float32)

    mask_depth1 = (depth_map > 1.0)
    ratio_11 = np.sum(mask_depth1) / (mask_depth1.shape[1] * mask_depth1.shape[2])
    depth_map[mask_depth1] = 0.0

    mask_1 = (depth_map > 1.0)
    num_1 = np.sum(mask_1)

    mask_depth2 = (depth_map < 0.0)
    ratio_12 = np.sum(mask_depth2) / (mask_depth2.shape[1] * mask_depth2.shape[2])
    depth_map[mask_depth2] = 0.0

    mask_2 = (depth_map < 0.0)
    num_2 = np.sum(mask_2)

    # Invert
    valid_pixels = (depth_map > 0.0)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel, iterations=2)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map == 0.0)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    empty_pixels = (depth_map == 0.0)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.0)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.0)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    return depth_map

p = r'/home/ubuntu/home/wck/GeoMVSNet_test_wwl_s1/D:\9054\depthmap.csv'
with open(p,encoding = 'utf-8') as f:
    data = np.loadtxt(f,delimiter = ",")
    data = data.reshape((1,512,640))


isnan = np.isnan(data)  # 判断每个元素是不是nan,返回[False,False,False,False,True]

print(True in isnan)
print(data[0][511][639])
fill_in_fast(data)