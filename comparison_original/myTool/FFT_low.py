import cv2
import numpy as np


# log2( row / half*2)==scale_factor
# log2(512 / 32*2) == 3
# row / (2^scale_factor) / 2 == half_bias
def FFT_low(img, scale_factor=4):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    half_bias = int(crow / scale_factor)
    # 低通滤波
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - half_bias:crow + half_bias, ccol - half_bias:ccol + half_bias] = 1

    # IDFT
    fshift = dft_shift * mask
    fshift = fshift[crow - half_bias:crow + half_bias, ccol - half_bias:ccol + half_bias]
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


def FFT_low_equal(img, scale_factor=4):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    half_bias = int(crow / scale_factor)
    # 低通滤波
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - half_bias:crow + half_bias, ccol - half_bias:ccol + half_bias] = 1

    # IDFT
    fshift = dft_shift * mask
    # fshift = fshift[crow - half_bias:crow + half_bias, ccol - half_bias:ccol + half_bias]
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back