import glob
import matplotlib

# matplotlib.use('Agg')
from skimage import transform
from skimage.transform import radon, iradon
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import warnings
import pywt

wavename = 'bior1.3'

warnings.filterwarnings('ignore')


class SourceImageDataset(Dataset):
    def __init__(self, labels_root, inputs_root):
        self.files_label = sorted(glob.glob("%s/*.png" % labels_root))
        self.files_input = sorted(glob.glob("%s/*.png" % inputs_root))

    def __getitem__(self, index):
        label_path = self.files_label[index % len(self.files_label)]
        normal = plt.imread(label_path)
        # label_name = label_path.split("\\")[-1]
        LL1, _ = pywt.dwt2(normal, wavename)
        label_LL2, _ = pywt.dwt2(LL1, wavename)
        label_LL3, _ = pywt.dwt2(label_LL2, wavename)
        label_LL2 = label_LL2[0:128, 0:128]
        label_LL3 = label_LL3[0:64, 0:64]

        input_path = self.files_input[index % len(self.files_input)]
        low = plt.imread(input_path)
        input_name = input_path.split("\\")[-1]
        # print(input_name)
        src_LL1, (LH, HL, HH) = pywt.dwt2(low, wavename)
        src_LL2, (LH, HL, HH) = pywt.dwt2(src_LL1, wavename)
        src_LL3, (LH, HL, HH) = pywt.dwt2(src_LL2, wavename)
        src_LL2 = src_LL2[0:128, 0:128]
        src_LL3 = src_LL3[0:64, 0:64]

        # normalized_low = (low - np.min(low)) / (np.max(low) - np.min(low))
        # normalized_normal = (normal - np.min(normal)) / (np.max(normal) - np.min(normal))
        # return normalized_low, normalized_normal, label_LL2, label_LL3, src_LL2, src_LL3, input_name
        return low, normal, label_LL2, label_LL3, src_LL2, src_LL3, input_name

    def __len__(self):
        return len(self.files_label)


class TargetImageDataset(Dataset):
    def __init__(self, inputs_root):
        self.files_low = sorted(glob.glob("%s/*.png" % inputs_root))

    def __getitem__(self, index):
        low = plt.imread(self.files_low[index % len(self.files_low)])
        LL1, (LH, HL, HH) = pywt.dwt2(low, wavename)
        LL2, (LH, HL, HH) = pywt.dwt2(LL1, wavename)
        LL3, (LH, HL, HH) = pywt.dwt2(LL2, wavename)
        LL2 = LL2[0:128, 0:128]
        LL3 = LL2[0:64, 0:64]

        # normalized_low = (low - np.min(low)) / (np.max(low) - np.min(low))
        # return normalized_low, LL2, LL3
        return low, LL2, LL3

    def __len__(self):
        return len(self.files_low)


if __name__ == '__main__':
    test_loader_source = DataLoader(
        # SourceImageDataset(labels_root='',
        SourceImageDataset(labels_root='',  #
                           inputs_root=rf''),  #
        batch_size=1,
        shuffle=False,
        drop_last=True,
        # sampler=DistributedSampler(ImageDataset)
    )
    len_test_loader_source = len(test_loader_source)
    print(len_test_loader_source)
    test_loader_source_iter = iter(test_loader_source)
    i = 0
    while i < len_test_loader_source:
        data_source = test_loader_source_iter.next()
        # inputs, labels, LL2, LL3, _, _ = data_source
        inputs, labels, _, _, LL2, LL3, input_name = data_source

        print(input_name)
        input_name, = input_name
        print(input_name)
        pass
