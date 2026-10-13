# cutoff是指除以几：
# 1就是直接最大最小归一化
# 2就是最大值设在中间归一化（从中间砍一刀）
import numpy as np


def myNormalization_error(array, cutoff=1):
    # 如果是多维就转成一维。
    oldShape = array.shape
    array = np.array(array.flatten())

    array = array.astype(float)  # 转换成浮点数，不然可能变01.
    max1, min1 = max(array), min(array)
    max1 = (max1 - min1) / cutoff
    if cutoff != 1:
        array.clip(max=max1)
    for i in range(len(array)):
        array[i] = round((array[i] - min1) / (max1 - min1), 5)

    array = array.reshape(oldShape)
    return array

