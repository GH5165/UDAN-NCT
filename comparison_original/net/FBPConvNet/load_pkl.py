import torch
from datasets import *
from model import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse
from math import sqrt
import time
import os

os.makedirs('', exist_ok=True)  #
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


total_s = 0  # ssim
total_p = 0  # psnr
total_r = 0  # rmse

test_loader = DataLoader(
    ImageDataset(labels_root='',  #
                 inputs_root=''),  #
    batch_size=1,
    shuffle=False,
    drop_last=True,
    num_workers=0
    # sampler=DistributedSampler(ImageDataset)
)

net = torch.load('')  #
net = net.cuda()
i = 1

if __name__ == '__main__':
    with torch.no_grad():
        for labels, inputs in test_loader:

            labels = labels.squeeze(0)
            labels = labels.cpu().numpy()

            # Generate image
            inputs = inputs.unsqueeze(-3)
            inputs = inputs.cuda()
            inputs = inputs.type(torch.cuda.FloatTensor)

            time_start = time.time()
            gen_imgs = net(inputs)
            time_end = time.time()
            with open('', "a") as a:  #
                a.write("|num: %s| |time: %f |\r\n"
                        % (i, (time_end - time_start)))
                a.close()
            gen_imgs = gen_imgs.squeeze(-3)
            gen_imgs = gen_imgs.squeeze(0)
            a = str(i).zfill(6)
            save_image(gen_imgs, '' % a, normalize=False)  #
            gen_imgs = gen_imgs.cpu().numpy()

            # print('labels type:', labels.__class__)
            # print('gen_imgs type:', gen_imgs.__class__)
            s = ssim(labels, gen_imgs)  # (img_true, img_test)
            # print(s.__class__)
            p = psnr(labels, gen_imgs)
            # print(p.__class__)
            # a = p.item()

            r = sqrt(mse(labels, gen_imgs))
            # print(m.__class__)
            with open('', "a") as a:  #
                a.write("|num: %s| |SSIM: %s | |PSNR: %s| |RMSE: %s |\r\n"
                        % (i, s, p, r))
                a.close()
            i = i+1
            total_s += float(s.item())
            total_p += float(p.item())
            total_r += float(r)
    with open('', "a") as f:  #
        f.write("|Average SSIM: %f | |Average PSNR: %f| |Average RMSE: %f |\r\n"
                % (total_s/len(test_loader), total_p/len(test_loader),
                   total_r/len(test_loader)))
        f.close()