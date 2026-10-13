import glob
import matplotlib

# matplotlib.use('Agg')
import torchvision
from skimage import transform
from skimage.transform import radon, iradon
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import warnings
import pywt


wavename = 'bior1.3'

warnings.filterwarnings('ignore')


class SourceImageDataset(Dataset):
    def __init__(self, labels_root, inputs_root):
        self.files_normal = sorted(glob.glob("%s/*.png" % labels_root))
        self.files_low = sorted(glob.glob("%s/*.png" % inputs_root))
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __getitem__(self, index):
        normal = plt.imread(self.files_normal[index % len(self.files_normal)])
        coeffs2_normal = pywt.dwt2(normal, wavename)
        n_LL1, (LH, HL, HH) = coeffs2_normal
        coeffs2_normal = pywt.dwt2(n_LL1, wavename)
        n_LL2, (LH, HL, HH) = coeffs2_normal
        coeffs2_normal = pywt.dwt2(n_LL2, wavename)
        n_LL2 = n_LL2[0:128, 0:128]
        n_LL3, (LH, HL, HH) = coeffs2_normal
        n_LL3 = n_LL3[0:64, 0:64]

        low = plt.imread(self.files_low[index % len(self.files_low)])
        l_LL1, (LH, HL, HH) = pywt.dwt2(low, wavename)
        l_LL2, (LH, HL, HH) = pywt.dwt2(l_LL1, wavename)
        l_LL3, (LH, HL, HH) = pywt.dwt2(l_LL2, wavename)
        l_LL2 = l_LL2[0:128, 0:128]
        l_LL3 = l_LL3[0:64, 0:64]

        # low = myNormalization(low)
        # l_LL2 = myNormalization(l_LL2)
        # l_LL3 = myNormalization(l_LL3)
        # normal = myNormalization(normal)
        # n_LL2 = myNormalization(n_LL2)
        # n_LL3 = myNormalization(n_LL3)
        return self.trans(low), self.trans(l_LL2), self.trans(l_LL3), self.trans(normal), self.trans(n_LL2), self.trans(
            n_LL3)

    def __len__(self):
        return len(self.files_normal)


class TargetImageDataset(Dataset):
    def __init__(self, inputs_root):
        self.files_low = sorted(glob.glob("%s/*.png" % inputs_root))
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __getitem__(self, index):
        low = plt.imread(self.files_low[index % len(self.files_low)])
        coeffs2_low = pywt.dwt2(low, wavename)
        LL1, (LH, HL, HH) = coeffs2_low
        coeffs2_low = pywt.dwt2(LL1, wavename)
        LL2, (LH, HL, HH) = coeffs2_low
        coeffs2_low = pywt.dwt2(LL2, wavename)
        LL2 = LL2[0:128, 0:128]
        LL3, (LH, HL, HH) = coeffs2_low
        LL3 = LL2[0:64, 0:64]

        # low = myNormalization(low)
        # LL2 = myNormalization(LL2)
        # LL3 = myNormalization(LL3)
        return self.trans(low), self.trans(LL2), self.trans(LL3)

    def __len__(self):
        return len(self.files_low)


if __name__ == '__main__':
    dataset = SourceImageDataset(
        labels_root=r"D:\gh\pycharmProject\pytorch\lowdoes_MMD\data\Cat2_real_norm_noise\train\labels_raw",
        inputs_root=r"D:\gh\pycharmProject\pytorch\lowdoes_MMD\data\Cat2_real_norm_noise\train\labels")
    for low, l2, l3, normal, n2, n3 in dataset:
        print(low.shape)
        print(l2.shape)
        print(l3.shape)
        print(normal.shape)
        print(n2.shape)
        print(n3.shape)
