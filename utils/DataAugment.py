import numpy as np
import cv2
import torch
from torch import nn
from torch import Tensor
import random


class Augment:
    def __init__(self, augments: list = []):
        self.augments = augments

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for ag in self.augments:
            img = ag(img)
        return img


class Salt_Pepper_Noise:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image: np.ndarray):
        """
        添加椒盐噪声
        :param image: 输入图像
        :param prob: 噪声比
        :return: 带有椒盐噪声的图像
        """
        thres = 1 - self.prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = np.random.rand()
                if rdn < self.prob:
                    image[i, j] = 0
                elif rdn > thres:
                    image[i, j] = 255
        return image


class Width_Shift_Range:
    def __init__(self, rate):
        super(Width_Shift_Range, self).__init__()
        self.rate = rate

    def __call__(self, img):
        rate = np.random.uniform(0, self.rate)
        if len(img.shape) == 2:
            h, w = img.shape
        else:
            h, w, c = img.shape
        x = int(w * rate)  # 计算平移像素
        if np.random.rand() < 0.5:  # 随机左右平移
            x = -x
        M = np.float32([[1, 0, x], [0, 1, 0]])
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return shifted


class Gaussian_Noise:
    def __init__(self, means, sigma, percetage):
        self.means = means
        self.sigma = sigma
        self.percetage = percetage

    def __call__(self, src):
        NoiseImg = src
        NoiseNum = int(self.percetage * src.shape[0] * src.shape[1])
        for i in range(NoiseNum):
            # 每次取一个随机点
            # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
            # random.randint生成随机整数
            # 高斯噪声图片边缘不处理，故-1
            randX = random.randint(0, src.shape[0] - 1)
            randY = random.randint(0, src.shape[1] - 1)
            # 此处在原有像素灰度值上加上随机数
            NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(self.means, self.sigma)
            # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
            if NoiseImg[randX, randY] < 0:
                NoiseImg[randX, randY] = 0
            elif NoiseImg[randX, randY] > 255:
                NoiseImg[randX, randY] = 255
        return NoiseImg


class Height_Shift_Range:
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, img):
        rate = np.random.uniform(0, self.rate)
        if len(img.shape) == 2:
            h, w = img.shape
        else:
            h, w, c = img.shape
        y = int(h * rate)  # 计算平移像素
        if np.random.rand() < 0.5:  # 随机左右平移
            y = -y
        M = np.float32([[1, 0, 0], [0, 1, y]])
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return shifted

#
# aug=Augment()
# img=cv2.imread("../1.jpeg",0)
# res=aug(img)
# cv2.imshow("",res)
# cv2.waitKey(0)
