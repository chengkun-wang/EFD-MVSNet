import numpy as np
import cv2
from PIL import Image
from pylab import *


class LBP:
    def __init__(self):
        # uniform_map为等价模式的58种特征值从小到大进行序列化编号得到的字典  58种
        self.uniform_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 12: 8, 14: 9, 15: 10, 16: 11, 24: 12,
                            28: 13, 30: 14, 31: 15, 32: 16, 48: 17, 56: 18, 60: 19, 62: 20, 63: 21, 64: 22, 96: 23,
                            112: 24, 120: 25, 124: 26, 126: 27, 127: 28, 128: 29, 129: 30, 131: 31, 135: 32, 143: 33,
                            159: 34, 191: 35, 192: 36, 193: 37, 195: 38, 199: 39, 207: 40, 223: 41, 224: 42, 225: 43,
                            227: 44, 231: 45, 239: 46, 240: 47, 241: 48, 243: 49, 247: 50, 248: 51, 249: 52, 251: 53,
                            252: 54, 253: 55, 254: 56, 255: 57}
        self.LBP_MASK = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2}

    # 图像的LBP原始特征计算算法：将图像指定位置的像素与周围8个像素比较
    # 比中心像素大的点赋值为1，比中心像素小的赋值为0，返回得到的二进制序列
    def calute_basic_lbp(self, image_array, i, j):
        sum = []
        num = 0
        if image_array[i - 1, j - 1] != image_array[i, j]:
            sum.append(1)
            num += 1
        else:
            sum.append(0)
        if image_array[i - 1, j] != image_array[i, j]:
            sum.append(1)
            num += 1
        else:
            sum.append(0)
        if image_array[i - 1, j + 1] != image_array[i, j]:
            sum.append(1)
            num += 1
        else:
            sum.append(0)
        if image_array[i, j - 1] != image_array[i, j]:
            sum.append(1)
            num += 1
        else:
            sum.append(0)
        if image_array[i, j + 1] != image_array[i, j]:
            sum.append(1)
            num += 1
        else:
            sum.append(0)
        if image_array[i + 1, j - 1] != image_array[i, j]:
            sum.append(1)
            num += 1
        else:
            sum.append(0)
        if image_array[i + 1, j] != image_array[i, j]:
            sum.append(1)
            num += 1
        else:
            sum.append(0)
        if image_array[i + 1, j + 1] != image_array[i, j]:
            sum.append(1)
            num += 1
        else:
            sum.append(0)
        return sum, num

    # 获取值r的二进制中1的位数
    def calc_sum(self, r):
        num = 0
        while (r):
            r &= (r - 1)
            num += 1
        return num

    # 获取图像的LBP原始模式特征
    def lbp_basic(self, image_array):
        mask = {}
        basic_array = np.zeros(image_array.shape, np.uint8)
        width = image_array.shape[0]
        height = image_array.shape[1]
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                num = self.calute_basic_lbp(image_array, i, j)[1]
                # sum = num * 31
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
        mask = {"0": mask0,
                "1": np.logical_or.reduce([mask1, mask2, mask3, mask4, mask5]),
                "2": np.logical_or.reduce([mask6, mask7, mask8])}
        print(np.sum(mask0 == True) / (1200 * 1600))
        print(np.sum(mask1 == True) / (1200 * 1600))
        print(np.sum(mask2 == True) / (1200 * 1600))
        print(np.sum(mask3 == True) / (1200 * 1600))
        print(np.sum(mask4 == True) / (1200 * 1600))
        print(np.sum(mask5 == True) / (1200 * 1600))
        print(np.sum(mask6 == True) / (1200 * 1600))
        print(np.sum(mask7 == True) / (1200 * 1600))
        print(np.sum(mask8 == True) / (1200 * 1600))
        # print(np.sum(mask2 == True))
        return mask

    # 获取图像的LBP等价模式特征
    def lbp_uniform(self, image_array):
        uniform_array = np.zeros(image_array.shape, np.uint8)
        basic_array = self.lbp_basic(image_array)
        width = image_array.shape[0]
        height = image_array.shape[1]
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                k = basic_array[i, j] << 1
                if k > 255:
                    k = k - 255
                xor = basic_array[i, j] ^ k  # ^是异或运算符，用于按位比较两个二进制数字，如果对应位上的数字不同，则该位的运算结果为1，否则为0
                num = self.calc_sum(xor)
                if num <= 2:
                    uniform_array[i, j] = self.uniform_map[basic_array[i, j]]
                else:
                    uniform_array[i, j] = 58
        return uniform_array

    # 绘制指定维数和范围的图像灰度归一化统计直方图
    def show_hist(self, img_array, im_bins, im_range):
        # 直方图的x轴是灰度值,y轴是图片中具有同一个灰度值的点的数目, [img_array]原图像图像格式为uint8或ﬂoat32,[0]0表示灰度,None无掩膜
        hist = cv2.calcHist([img_array], [0], None, im_bins, im_range)
        hist = cv2.normalize(hist, hist).flatten()  # 对计算出来的直方图数据进行归一化处理，并将结果扁平化为1D数组
        # print(hist)
        plt.plot(hist, color='r')
        plt.xlim(im_range)  # 设置X轴的取值范围
        plt.show()

    # 绘制图像原始LBP特征的归一化统计直方图
    def show_basic_hist(self, img_array):
        self.show_hist(img_array, [256], [0, 256])

    # 绘制图像等价模式LBP特征的归一化统计直方图
    def show_uniform_hist(self, img_array):
        self.show_hist(img_array, [60], [0, 60])  # [60]把像素值区间分成60部分，[0, 60]像素值区间

    # 显示图像
    def show_image(self, image_array):
        plt.imshow(image_array, cmap='Greys_r')
        plt.show()

    def read_img(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32) / 255.
        gray_image = img.convert('L')
        gray_image = np.array(gray_image, dtype=np.float32)
        return np_img, gray_image


if __name__ == '__main__':
    # image = cv2.imread('rect_001_0_r5000.png')
    # image_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # BGR格式图像转灰度图像
    lbp = LBP()
    np_img, gray_img = lbp.read_img('rect_001_0_r5000.png')
    # plt.imshow(gray_img, cmap='Greys_r')  # 去掉参数就是热量图了
    # plt.title('Original')
    # plt.show()

    # 获取图像原始LBP特征，并显示其统计直方图与特征图像
    mask = lbp.lbp_basic(gray_img)
    # print(mask)
    print(np.sum(mask["0"] == True) / (1200 * 1600))
    print(np.sum(mask["1"] == True) / (1200 * 1600))
    print(np.sum(mask["2"] == True) / (1200 * 1600))


