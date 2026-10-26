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
from myTool.light_norm import aug
from myTool.mkdir import mkdir
from myTool.myNormalization import myNormalization

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# in_folder = ''
in_folder = ''  # TODO: set local path
netName = "FBPConvNet-nd_ndct"
# netName = "REDCNN-nd_ndct"
# netName = "DnCNN-nd_ndct"

modelsPath = "./saved_models"
mkdir(modelsPath)
evaluationsPath = "./saved_evaluations"
mkdir(evaluationsPath)
mkdir(modelsPath + f"/{netName}")
mkdir(evaluationsPath + f"/{netName}")

batch_size = 1
test_loader = DataLoader(
    # dataset=ImageDataset(inputs_root='',
    dataset=ImageDataset(inputs_root=in_folder,
                         labels_root=''),  # TODO: set local path
    batch_size=batch_size,
    shuffle=True,
    # num_workers=4
)

net = torch.load(modelsPath + f'/{netName}/{netName}_80.pkl')
# net = torch.load(modelsPath + f'/{netName}/best_model.pkl')
net = net.cuda()
# net = {key: net[key].cuda() for key in net}
i = 1
total_s = 0  # ssim
total_p = 0  # psnr
total_r = 0  # rmse
length = len(test_loader) * batch_size
if __name__ == '__main__':
    s_list, p_list, r_list = [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.eval()
    with torch.no_grad():
        for inputs, labels, labels_name in test_loader:
            # 解包dataloader时 会变成元组
            # inputs = inputs.to(device)
            inputs = inputs.to(device).float()
            netout = net(inputs)

            for img_out, label, label_name in zip(netout, labels, labels_name):
                img_out = img_out.squeeze(0)
                img_out = img_out.cpu().numpy()
                label = label.squeeze(0)
                label = label.cpu().numpy()

                # label = myNormalization(label)
                # img_out = myNormalization(img_out)

                s = ssim(label, img_out)
                s = float(s.item())
                p = psnr(label, img_out)
                p = float(p.item())
                r = sqrt(mse(label, img_out))
                s_list.append(s)
                p_list.append(p)
                r_list.append(r)

                total_s += float(s)
                total_p += float(p)
                total_r += float(r)

                # 存储
                mkdir(in_folder + "\\results")
                dstPath = in_folder + f"\\results\\{netName}"
                mkdir(dstPath)
                mkdir(f'{dstPath}\\mat')
                label_name = ''.join(label_name)
                # mat_name = labels_name[:-4] + ".mat"
                img = myNormalization(img_out) * 255
                img = Image.fromarray(img).convert("L")
                img.save(f"{dstPath}\\{label_name}")
                print(f"{dstPath}\\{label_name}")
                # sio.savemat(f'{dstPath}\\mat\\{mat_name}', {'img': img_out})

            i = i + 1
            with open(f"{dstPath}\\each_img_parameters.txt", "a") as a:
                a.write(f"|num: {i} | |name: {labels_name} | |SSIM: {s} | |PSNR: {p} | |RMSE: {r} |\n")
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
        with open(f"{dstPath}\\context_encoder_parameters.txt", "a") as f:
            f.write(
                f'|SSIM: {aver_SSIM} ± {std_SSIM} |PSNR: {aver_PSNR} ± {std_PSNR} |RMSE: {aver_RMSE} ± {std_RMSE} |\n')
