import argparse
from torch.optim.lr_scheduler import MultiStepLR
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse
from math import sqrt
from datasets import *
from model import *
import os
import warnings
from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# write = SummaryWriter(r"D:\neutron imaging\residual imaging\code\python\111")

os.makedirs(r"D:\neutron imaging\limited_img\Result\head_FBPConVnet", exist_ok=True)
os.makedirs(r"D:\neutron imaging\limited_img\Result\head_FBPConVnet/saved_models", exist_ok=True)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 参数定义
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="number of epochs of training")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--train_label_dataset", type=str, default="label", help="name of the train label dataset")
parser.add_argument("--train_inputs_dataset", type=str, default="ld_0.375", help="name of the train inputs dataset")
parser.add_argument("--test_label_dataset", type=str, default="label", help="name of the test label dataset")
parser.add_argument("--test_inputs_dataset", type=str, default="ld_0.375", help="name of the test inputs dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")  # 学习率
parser.add_argument('--momentum', type=float, default=0.99)
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=7, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args(args=[])
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Loss function
criterion = torch.nn.L1Loss()  # Recomstruction Loss

# Initialize net and discriminator
net = FBPCONVNet()

if cuda:
    net.cuda()
    criterion.cuda()


if opt.epoch != 0:
    # Load pretrained models
    net = torch.load(
        r"D:\neutron imaging\limited_img\Result\head_FBPConVnet/saved_models/FBPConvNet_%d.pkl" % opt.epoch)

else:
    # Initialize weights
    net.apply(weights_init_normal)


train_loader = DataLoader(
    ImageDataset(labels_root=r"D:\neutron imaging\limited_img\data\head\train\%s"
                             % opt.train_label_dataset,
                 inputs_root=r"D:\neutron imaging\limited_img\data\head\train\%s"
                             % opt.train_inputs_dataset),
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=opt.n_cpu
)
test_loader = DataLoader(
    ImageDataset(labels_root=r"D:\neutron imaging\limited_img\data\head\test\%s"
                             % opt.test_label_dataset,
                 inputs_root=r"D:\neutron imaging\limited_img\data\head\test\%s"
                             % opt.test_inputs_dataset),
    batch_size=1,
    shuffle=False,
    drop_last=True,
    num_workers=3
)

# Optimizers
optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum,
                                weight_decay=1e-8)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def train(epoch):
    train_g_loss = 0

    for i, (imgs, lose_degree_img) in enumerate(train_loader):

        # Configure input
        imgs = Variable(imgs.type(Tensor))
        lose_degree_img = Variable(lose_degree_img.type(Tensor))
        lose_degree_img = lose_degree_img.unsqueeze(-3)

        # if epoch == 0:
        #     write.add_graph(net, lose_degree_img)

        # training net work
        net.train()
        fake_img = net(lose_degree_img)
        imgs = imgs.unsqueeze(-3)

        # pixelwise loss
        g_pixel = criterion(fake_img, imgs)

        train_g_loss += float(g_pixel.item())

        optimizer.zero_grad()
        g_pixel.backward()
        optimizer.step()

        # print("|Epoch %d/%d| |Batch %d/%d| |G_loss: %f|"
        #       % (epoch+1, opt.n_epochs, i+1, len(train_loader), g_pixel.item()))
        with open(r"D:\neutron imaging\limited_img\Result\head_FBPConVnet/iteration_loss_0.375.txt", "a") as q:
            q.write("|Epoch %d/%d| |Batch %d/%d| |G_pixel: %f|\r\n"
                    % (epoch+1, opt.n_epochs, i+1, len(train_loader), g_pixel.item()))
            q.close()
    
    # scheduler_G.step()

    print("|Epoch %d/%d| |totla_G_loss: %f|"
          % (epoch+1, opt.n_epochs, train_g_loss/len(train_loader)))
    with open(r"D:\neutron imaging\limited_img\Result\head_FBPConVnet/total_loss_0.375.txt", "a") as p:
        p.write("|Epoch %d/%d| |totla_G_loss: %f|\r\n"
                % (epoch+1, opt.n_epochs, train_g_loss/len(train_loader)))
        p.close()


def test():
    total_s = 0  # ssim
    total_p = 0  # psnr
    total_r = 0  # rmse

    with torch.no_grad():
        for labels, inputs in test_loader:
            labels = labels.squeeze(0)
            labels = labels.cpu().numpy()

            # Generate image
            inputs = inputs.unsqueeze(-3)
            inputs = Variable(inputs.type(Tensor))
            net.eval()
            gen_imgs = net(inputs)
            gen_imgs = gen_imgs.squeeze(-3)
            gen_imgs = gen_imgs.squeeze(0)
            gen_imgs = gen_imgs.cpu().numpy()

            s = ssim(labels, gen_imgs)  # (img_true, img_test)
            p = psnr(labels, gen_imgs)
            r = sqrt(mse(labels, gen_imgs))

            total_s += float(s.item())
            total_p += float(p.item())
            total_r += float(r)

    print('|Epoch %d/%d| |Average SSIM: %f | |Average PSNR: %f| |Average RMSE: %f |'
          % ((epoch+1), opt.n_epochs, total_s/len(test_loader), total_p/len(test_loader),
                   total_r/len(test_loader)))
    with open(r"D:\neutron imaging\limited_img\Result\head_FBPConVnet/test_parameters_0.375.txt", "a") as f:
        f.write("|Epoch %d/%d| |Average SSIM: %f | |Average PSNR: %f| |Average RMSE: %f |\r\n"
                % ((epoch+1), opt.n_epochs, total_s/len(test_loader), total_p/len(test_loader),
                   total_r/len(test_loader)))
    f.close()
    return total_s/len(test_loader), total_p/len(test_loader),  total_r/len(test_loader)


if __name__ == '__main__':
    for epoch in range(opt.n_epochs):
        train(epoch)
        torch.save(net,
                   r"D:\neutron imaging\limited_img\Result\head_FBPConVnet\saved_models/FBPConvNet_%d.pkl" % (epoch + 1))
        epoch_s, epoch_p, epoch_r = test()




