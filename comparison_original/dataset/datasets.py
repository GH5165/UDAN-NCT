import glob

import cv2
import torchvision
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


class ImageDataset(Dataset):
    def __init__(self, inputs_root, labels_root):
        # self.files = sorted(glob.glob(f"{inputs_root}\\*.tif"))
        self.files = glob.glob(f"{inputs_root}\\*.png")
        # self.files_using = sorted(glob.glob(f"{labels_root}\\*.tif"))
        self.files_using = glob.glob(f"{labels_root}\\*.png")
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __getitem__(self, index):
        inputs = cv2.imread(self.files[index % len(self.files)], cv2.IMREAD_GRAYSCALE)
        # inputs = plt.imread(self.files[index % len(self.files)])
        labels = cv2.imread(self.files_using[index % len(self.files_using)], cv2.IMREAD_GRAYSCALE)
        # labels = plt.imread(self.files_using[index % len(self.files_using)])
        labels_path = self.files_using[index % len(self.files_using)]
        labels_name = labels_path.split("\\")[-1]

        return self.trans(inputs).float(), self.trans(labels).float(), labels_name

    def __len__(self):
        return len(self.files_using)


if __name__ == '__main__':
    a = ImageDataset(
        labels_root='',  #
        inputs_root='')  #
    for inputs, labels, name in a:
        print(inputs)
        print(labels)
