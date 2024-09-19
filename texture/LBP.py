import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

img1 = cv.imread("./rect_001_0_r5000.png", cv.IMREAD_COLOR)
plt.imshow(img1[:, :, ::-1])
plt.axis('off')
plt.show()
img_gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)


img2 = cv.imread("./rect_002_0_r5000.png", cv.IMREAD_COLOR)
plt.imshow(img2[:, :, ::-1])
plt.axis('off')
plt.show()
# 转换为灰度图
img_gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


img3 = cv.imread("./rect_003_0_r5000.png", cv.IMREAD_COLOR)
plt.imshow(img3[:, :, ::-1])
plt.axis('off')
plt.show()
img_gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
"""'default'：原始的局部二值模式，它是灰度但不是旋转不变的。
	'ror'：扩展灰度和旋转不变的默认实现。
	'uniform'：改进的旋转不变性和均匀的模式以及角度空间的更精细的量化，灰度和旋转不变。
	'nri_uniform'：非旋转不变的均匀图案变体，它只是灰度不变的R199。
	'VAR'：局部对比度的旋转不变方差度量，图像纹理是旋转但不是灰度不变的。"""
radius = 1
#   邻域像素点个数
n_points = 8 * radius
lbp_uniform1 = local_binary_pattern(img_gray1, n_points, radius, method="uniform")
lbp_uniform2 = local_binary_pattern(img_gray2, n_points, radius, method="uniform")
lbp_uniform3 = local_binary_pattern(img_gray3, n_points, radius, method="uniform")
# print(lbp_uniform)
plt.imshow(lbp_uniform1)
plt.axis('off')
plt.savefig('lbp_uniform1.png')
plt.show()

plt.imshow(lbp_uniform2)
plt.axis('off')
plt.savefig('lbp_uniform2.png')
plt.show()

plt.imshow(lbp_uniform3)
plt.axis('off')
plt.savefig('lbp_uniform3.png')
plt.show()
