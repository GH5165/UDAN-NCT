import argparse
import datetime
import os

import torch
from torch.utils.data import DataLoader

from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from math import sqrt

# =======================================程序设置====================================================
from net.RED_CNN.networks import RED_CNN
from dataset.datasets import ImageDataset
from myTool.mkdir import mkdir

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--upscale_factor", type=int, default=4, help="number of epochs of training")
# parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
'''。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。'''
# =============================网络设置：net, loss, optimizer========================================
# Losses
myLoss = torch.nn.MSELoss()
# net
myNet = RED_CNN(out_ch=96)
netName = "REDCNN-ld_ndct"
# data
train_loader = DataLoader(
    dataset=ImageDataset(inputs_root=r'D:\gh\pycharmProject\pytorch\WT_SR\data\full_dose\test1',
                         labels_root=r'D:\gh\pycharmProject\pytorch\WT_SR\data\low_dose\test1'),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    pin_memory=True,
)
test_loader1 = DataLoader(
    dataset=ImageDataset(labels_root=r"D:\gh\CodeMATLAB\result\FDK\FDK_raw_200Views",
                         inputs_root=r"D:\gh\CodeMATLAB\result\FDK\FDK_raw_200Views_noise1e4"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    pin_memory=True,
)

modelsPath = "./saved_models"
mkdir(modelsPath)
evaluationsPath = "./saved_evaluations"
mkdir(evaluationsPath)
mkdir(modelsPath + f"/{netName}")
mkdir(evaluationsPath + f"/{netName}")

# cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myNet = myNet.to(device)
myLoss = myLoss.to(device)

if opt.epoch != 0:
    # Load pretrained models
    myNet = torch.load(modelsPath + f"/{netName}/{netName}_{opt.epoch}.pkl")

# Optimizers
optimizer = torch.optim.Adam(myNet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)  # 每十次迭代，学习率减半

# Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
'''。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。'''


def train(epoch):
    sum_loss = 0
    length = len(train_loader) * opt.batch_size
    for i, data in enumerate(train_loader):
        # 一个batch
        inputs, labels, labels_name = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        myNet.train()
        netout = myNet(inputs)

        # Total loss
        gloss = myLoss(netout, labels)
        gloss.backward()
        optimizer.step()
        # --------------
        #  Log Progress
        # --------------
        sum_loss += gloss.item()
        with open(evaluationsPath + f"/{netName}/loss.txt", "a") as f1:
            # f1.write(f"datetime.datetime.now():")
            f1.write(f"[Epoch {epoch + 1}/{opt.n_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {gloss.item()} ]\n")

    aver_loss = sum_loss / length
    with open(evaluationsPath + f"/{netName}/aver_loss.txt", "a") as f2:
        f2.write(f"[Epoch {epoch + 1}/{opt.n_epochs}]\t aver_loss: {aver_loss} \n")


def test1():
    total_s = 0  # ssim
    total_p = 0  # psnr
    total_r = 0  # rmse
    s_list, p_list, r_list = [], [], []
    length = len(test_loader1) * opt.batch_size

    with torch.no_grad():
        for inputs, labels, labels_name in test_loader1:
            # optimizer.zero_grad()
            myNet.eval()
            inputs = inputs.to(device)
            netout = myNet(inputs)

            # for img_out in netout:
            for img_out, label, label_name in zip(netout, labels, labels_name):
                img_out = img_out.squeeze(0)
                img_out = img_out.cpu().numpy()
                label = label.squeeze(0)
                label = label.cpu().numpy()

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

    print(
        f'Epoch {epoch + 1}/{opt.n_epochs}|SSIM: {aver_SSIM} ± {std_SSIM} |PSNR: {aver_PSNR} ± {std_PSNR} |RMSE: {aver_RMSE} ± {std_RMSE} |')

    with open(evaluationsPath + f"/{netName}/test_parameters.txt", "a") as f:
        f.write(f"{datetime.datetime.now()}:")
        f.write(
            f'Epoch {epoch + 1}/{opt.n_epochs}|SSIM: {aver_SSIM} ± {std_SSIM} |PSNR: {aver_PSNR} ± {std_PSNR} |RMSE: {aver_RMSE} ± {std_RMSE} |')
        f.write('\n')
    return aver_SSIM, aver_PSNR, aver_RMSE


if __name__ == '__main__':
    max_SSIM = 0
    begin_time = datetime.datetime.now()
    print("开始：", begin_time)

    for epoch in range(opt.epoch, opt.n_epochs):
        # 一个epoch
        train(epoch)
        print("训练一个Epoch后:", datetime.datetime.now())
        aver_SSIM, aver_PSNR, aver_RMSE = test1()
        print(f"测试一个epoch后：{datetime.datetime.now()}")
        # Save model checkpoints
        # torch.save(net.state_dict(), "saved_models/generator_%d.pkl" % epoch)
        if aver_SSIM > max_SSIM:
            max_SSIM = aver_SSIM
            torch.save(myNet, modelsPath + f"/{netName}/best_model.pkl")
        if (epoch + 1) % 20 == 0:
            torch.save(myNet, modelsPath + f"/{netName}/{netName}_{epoch + 1}.pkl")

    print(netName)
