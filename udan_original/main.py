import argparse

import sys

import torch.optim as optim

import torch

from torch.utils.data import DataLoader



import model3

import model6

from datasets import *

# import model

import os

import warnings

# from test_liver import test_liver

from other.tool.mkdir import mkdir

from test_chest import test_chest

from torch.autograd import Variable

# from test4 import test4

import mmd



GAMMA = 10 ^ 3





def mmd_loss(x_src, x_tar):

    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])

    # return mmd.linear_mmd2(x_src, x_tar)





warnings.filterwarnings('ignore')



os.makedirs(r".\loss_data", exist_ok=True)



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# 参数定义

parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=0, help="start of epoch of training")

parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")

parser.add_argument("--features", type=int, default=256, help="features")

parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")

parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")

parser.add_argument("--source_label_dataset", type=str, default="source", help="name of the source label dataset")

parser.add_argument("--source_inputs_dataset", type=str, default="source", help="name of the source inputs dataset")

# parser.add_argument("--test_label_dataset", type=str, default="test", help="name of the test label dataset")

parser.add_argument("--target_inputs_dataset", type=str, default="target", help="name of the target inputs dataset")

parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")  # 学习率

parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")

parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")

parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")

parser.add_argument("--channels", type=int, default=1, help="number of image channels")

# parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")  # 图像的抽样间隔

opt = parser.parse_args(args=[])

# print(opt)



batch_size = opt.batch_size



model_root = 'models'

source_dataset_name = 'source_img'

target_dataset_name = 'target_img'

mkdir(r".\saved_models")

mkdir(r".\saved_evaluations")



cuda = True if torch.cuda.is_available() else False





def weights_init_normal(m):

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:

        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm2d") != -1:

        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)

        torch.nn.init.constant_(m.bias.data, 0.0)





# Loss function

loss_denoise = torch.nn.MSELoss()

# netName = 'noise2_model4_current.pkl'   # 删除DBPN且删除跳跃连接。直接编码解码

# netName = 'noise2_model3new_current.pkl'  # 删除DBPN，小波+dncnn代替跳跃连接

netName = 'noise2_model6_current.pkl'  # model3（最优）基础上删掉MMD

# netName = 'noise2_model5_current.pkl'  # 删除DBPN但不删除跳跃连接



source_loader = DataLoader(

    SourceImageDataset(

        # labels_root='',
        # labels_root='',
        labels_root='',  #
        # inputs_root=''),
        # inputs_root=''),
        inputs_root=''),  #
    batch_size=opt.batch_size,

    shuffle=True,

    drop_last=True,

    num_workers=opt.n_cpu

    # sampler=DistributedSampler(ImageDataset)

)



target_loader = DataLoader(

    # TargetImageDataset(inputs_root=rf''),
    # TargetImageDataset(inputs_root=rf''),
    TargetImageDataset(inputs_root=rf''),  #
    batch_size=opt.batch_size,

    shuffle=True,

    drop_last=True,

    num_workers=opt.n_cpu

    # sampler=DistributedSampler(ImageDataset)

)

# load model

if opt.epoch != 0:

    my_net = torch.load(f'{netName}')

else:

    # my_net = model3.UNet(n_channels=1, n_classes=1)

    my_net = model6.UNet(n_channels=1, n_classes=1)

    # my_net = model2.UNet2(n_channels=1, n_classes=1)

# my_net = model.UNet(n_channels=1, n_classes=1)



# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=opt.lr)



if cuda:

    my_net = my_net.cuda()

    loss_denoise = loss_denoise.cuda()



for p in my_net.parameters():

    p.requires_grad = True



# training

best_accu_t = 0.0

n_epoch = opt.n_epochs

max_SSIM = 0

for epoch in range(opt.epoch, n_epoch):



    len_dataloader = min(len(source_loader), len(target_loader))

    # data_source_iter = iter(source_loader)

    # data_target_iter = iter(target_loader)



    total_loss = 0

    total_mmd_loss = 0



    # for i in range(len_dataloader):

    for (i, data_source), (j, data_target) in zip(enumerate(source_loader),

                                                  enumerate(target_loader)):



        # training model using source data

        # data_source = data_source_iter.next()

        # s_img, s_label, s_LL2, s_LL3, _, _ = data_source

        s_input, s_label, _, _, s_LL2, s_LL3, input_name = data_source



        # data_target = data_target_iter.next()

        t_img, _, _ = data_target



        s_input = Variable(torch.unsqueeze(s_input, dim=1).float(), requires_grad=False)

        s_label = Variable(torch.unsqueeze(s_label, dim=1).float(), requires_grad=False)

        t_img = Variable(torch.unsqueeze(t_img, dim=1).float(), requires_grad=False)

        s_LL2 = Variable(torch.unsqueeze(s_LL2, dim=1).float(), requires_grad=False)

        s_LL3 = Variable(torch.unsqueeze(s_LL3, dim=1).float(), requires_grad=False)



        my_net.zero_grad()



        if cuda:

            s_input = s_input.cuda()

            s_label = s_label.cuda()

            t_img = t_img.cuda()

            s_LL2 = s_LL2.cuda()

            s_LL3 = s_LL3.cuda()

            # s_LL7 = s_LL7.cuda()

            # s_LL8 = s_LL8.cuda()



        # s_img = torch.tensor.float(s_img)



        s_img_output, s_coder, t_coder = my_net(x_src=s_input, x_tar=t_img, LL2=s_LL2, LL3=s_LL3)



        err_s_denoise = loss_denoise(s_img_output, s_label)



        loss_mmd = 0



        for j in range(batch_size):

            for k in range(1024):

                loss_mmd += mmd_loss(s_coder[j, k, :, :], t_coder[j, k, :, :])

        loss_mmd = loss_mmd / (batch_size * opt.features)

        total_mmd_loss += loss_mmd.data.cpu().numpy()



        err = err_s_denoise + loss_mmd

        # err = err_s_denoise

        # err = err_s_denoise

        err.backward()

        optimizer.step()



        total_loss += err_s_denoise.data.cpu().numpy()



        # err_s_label.data.cpu().numpy()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, mmd: %f'\

                         % ((epoch + 1), i + 1, len_dataloader, err.data.cpu().numpy(), loss_mmd.data.cpu().numpy()))

        sys.stdout.flush()

        with open(rf".\saved_evaluations\train_iter_parameters_{netName[:-12]}.txt",

                  "a") as f:

            f.write("|Epoch %d/%d| |iter: %d / all %d| |Loss: %f | |mmd: %f |\n"

                    % ((epoch + 1), opt.n_epochs, i + 1, len_dataloader, err.data.cpu().numpy(),

                       loss_mmd.data.cpu().numpy()))



        torch.save(my_net, f'{netName}')



    print('\n')

    total_loss = total_loss / len_dataloader

    total_mmd_loss = total_mmd_loss / len_dataloader

    sys.stdout.write('\r epoch: %d, total_loss_avg: %f, total_mmd_loss_avg: %f'\

                     % ((epoch + 1), total_loss, total_mmd_loss))

    torch.cuda.empty_cache()

    print('\n')

    with open(fr".\saved_evaluations\train_parameters_{netName[:-12]}.txt",

              "a") as f:

        f.write("|Epoch %d/%d| |Loss: %f | |mmd: %f |\n"

                % ((epoch + 1), opt.n_epochs, total_loss, total_mmd_loss))

    f.close()



    aver_s, aver_r, aver_p = test_chest(epoch, f'{netName}', batch_size, opt.n_epochs)



    if aver_s > max_SSIM:

        max_SSIM = aver_s

        torch.save(my_net, f"./saved_models/{netName[:-12]}_best{epoch + 1}.pkl")

