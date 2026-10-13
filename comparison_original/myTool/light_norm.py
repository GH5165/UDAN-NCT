import os

import numpy as np
import cv2

from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from math import sqrt

from myTool.myNormalization import myNormalization


def compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""

    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def aug(src):
    """图像亮度增强"""
    if get_lightness(src) > 130:
        print("图片亮度足够，不做增强")
    # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
    # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。

    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # 去掉分位值区间之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)

    return out


def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness


if __name__ == '__main__':
    # result = cv2.imread(r'D:\gh\CodeMATLAB\result\FDK\FDK_raw_200Views_noise1e3_norm\results\FBPConvNet-nd_ndct\31.png')
    # label = cv2.imread(r'D:\gh\CodeMATLAB\result\FDK\FDK_raw_200Views_norm\31.png')
    # result = aug(result)
    # label = aug(label)
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    # result = myNormalization(result)
    # label = myNormalization(label)
    #
    # s = ssim(label, result)
    # s = float(s.item())
    # p = psnr(label, result)
    # p = float(p.item())
    # r = sqrt(mse(label, result))
    #
    # print(s, p, r)

    root_path = r"D:\gh\CodeMATLAB\result\FDK\FDK_raw_200Views_noise1e4\results\FBPConvNet-nd_ndct_norm"
    root_path = r"D:\gh\CodeMATLAB\result\FDK\FDK_raw_200Views_noise1e4\results\DnCNN-ld_ndct_norm"
    root_path = r"D:\gh\CodeMATLAB\result\FDK\FDK_raw_200Views_noise1e4\results\REDCNN-ld_ndct_norm"
    # root_path = r"D:\gh\CodeMATLAB\result\SART\SART_raw_100Views_norm"
    for file in os.listdir(root_path):
        in_path = root_path + f"\\{file}"
        # img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(in_path)
        img = aug(img)
        print(in_path)

        imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(in_path, imGray)
