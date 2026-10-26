import cv2
import torch
from PIL import Image

from torch.utils.data import DataLoader
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from math import sqrt
import scipy.io as sio
import os

from dataset.datasets import ImageDataset
from myTool.mkdir import mkdir
from myTool.myNormalization import myNormalization

# label_folder = ''
label_folder = ''  #
# netName = "FBPConvNet-ld_ndct"
# netName = "REDCNN-ld_ndct"
# netName = "DnCNN-ld_ndct"  #
# in_folder = f''
# netName = "SART"  #
# in_folder = f''
netName = "noise2_model3_best41"
in_folder = f''  #
# netName = "self"
# in_folder = label_folder

if __name__ == '__main__':

    total_s = 0  # ssim
    total_p = 0  # psnr
    total_r = 0  # rmse
    length = len(os.listdir(in_folder))
    for label_name, input_name in zip(os.listdir(label_folder), os.listdir(in_folder)):
        label = cv2.imread(label_folder + "\\" + label_name, cv2.IMREAD_GRAYSCALE)
        input = cv2.imread(in_folder + "\\" + input_name, cv2.IMREAD_GRAYSCALE)
        s_list, p_list, r_list = [], [], []

        input = myNormalization(input)
        label = myNormalization(label)

        s = ssim(input, label)
        s = float(s.item())
        p = psnr(input, label)
        p = float(p.item())
        r = sqrt(mse(input, label))
        s_list.append(s)
        p_list.append(p)
        r_list.append(r)

        total_s += float(s)
        total_p += float(p)
        total_r += float(r)

    aver_SSIM = total_s / length
    aver_PSNR = total_p / length
    aver_RMSE = total_r / length

    s_sum, p_sum, r_sum = 0, 0, 0
    for s, p, r in zip(s_list, p_list, r_list):
        s_sum = s_sum + (s - aver_SSIM) ** 2
        p_sum = p_sum + (p - aver_PSNR) ** 2
        r_sum = r_sum + (r - aver_RMSE) ** 2
    std_SSIM = sqrt(s_sum / length)
    std_PSNR = sqrt(p_sum / length)
    std_RMSE = sqrt(r_sum / length)
    print(f'{netName}_norm |SSIM: {aver_SSIM} ± {std_SSIM} |PSNR: {aver_PSNR} ± {std_PSNR} |RMSE: {aver_RMSE} ± {std_RMSE} |\n')
