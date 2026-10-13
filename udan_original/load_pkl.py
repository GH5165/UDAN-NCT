import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# from skimage.measure import compare_ssim as ssim
from math import sqrt
import numpy as np
from torch.utils.data import DataLoader
from datasets import *
import scipy.io as sio

from other.tool.mkdir import mkdir
from other.tool.myNormalization import myNormalization

# modelroot = r'noise2_model4_current.pkl'  # nd
# modelroot = r'noise2_model3_current.pkl'  # ld-nd
# data_name = modelroot[:-4]
# modelroot = r'saved_models\noise2_model4_best24.pkl'  # ndct+1e4
# modelroot = r'saved_models\noise2_model3_best41.pkl'  # ndct+1e4
# modelroot = r'saved_models\noise2_model6_best38.pkl'  # ndct+1e4
modelroot = r'saved_models\noise2_model5_best12.pkl'  # ndct+1e4
my_net = torch.load(modelroot)
my_net.eval()
data_name = modelroot[13:-4]
mkdir(rf".\result\{data_name}")

batch_size = 1
epoch = 0
epoch_total = 200
total_s_source = 0  # ssim
total_p_source = 0  # psnr
total_r_source = 0  # rmse

# model_root = 'models'
# image_root = os.path.join('dataset', dataset_name)

cuda = True
cudnn.benchmark = True

"""load data"""
# data_name = r'FDK_raw_50Views'
test_loader_source = DataLoader(
    # SourceImageDataset(labels_root=r"D:\gh\CodeMATLAB\result\FDK\FDK_raw_200Views",
    SourceImageDataset(labels_root=r"E:\CodeMATLAB\6248R\CodeMATLAB\result\FDK\FDK_raw_200Views",
                       inputs_root=fr"E:\CodeMATLAB\6248R\CodeMATLAB\result\FDK\FDK_raw_200Views_noise1e4"),
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    # sampler=DistributedSampler(ImageDataset)
)

""" test """
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

len_test_loader_source = len(test_loader_source)
test_loader_source_iter = iter(test_loader_source)

std_all1 = np.zeros(len_test_loader_source * batch_size)
std_all2 = np.zeros(len_test_loader_source * batch_size)
std_all3 = np.zeros(len_test_loader_source * batch_size)

n_total_source = 0
i = 0
total_time = 0
for i, data_source in enumerate(test_loader_source):

    with torch.no_grad():
        # data_source = test_loader_source_iter.next()
        inputs, labels, _, _, LL2, LL3, input_name = data_source
        input_name, = input_name
        inputs = Variable(torch.unsqueeze(inputs, dim=1).float(), requires_grad=False)
        labels = Variable(torch.unsqueeze(labels, dim=1).float(), requires_grad=False)
        LL2 = Variable(torch.unsqueeze(LL2, dim=1).float(), requires_grad=False)
        LL3 = Variable(torch.unsqueeze(LL3, dim=1).float(), requires_grad=False)
        # labels = labels.cpu().numpy()

        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            LL2 = LL2.cuda()
            LL3 = LL3.cuda()

        # start = time.process_time()
        # netout, _, _ = my_net(inputs, inputs, LL2, LL3)
        # end = time.process_time()
        # during = end - start
        # total_time += during
        # print(during)
        start_time = time.time()
        netout, _, _ = my_net(inputs, inputs, LL2, LL3)
        end_time = time.time()
        during = end_time - start_time
        print("运行时间：", during, "秒")
        total_time += during

        netout = netout.squeeze(1)
        labels = labels.squeeze(1)

        netout = netout.cpu().numpy()
        labels = labels.cpu().numpy()
        for j in range(batch_size):
            # gen_img = myNormalization(netout[j, :, :])
            gen_img = netout[j, :, :]
            # label_img = myNormalization(labels[j, :, :])
            label_img = labels[j, :, :]
            # sio.savemat(name, {'img': gen_img})
            s = ssim(label_img, gen_img, data_range=1)  # (img_true, img_test)
            p = psnr(label_img, gen_img, data_range=1)
            r = sqrt(mse(label_img, gen_img))
            total_s_source += float(s)
            std_all1[i * batch_size + j] = float(s)
            total_p_source += float(p)
            std_all2[i * batch_size + j] = float(p)
            total_r_source += float(r)
            std_all3[i * batch_size + j] = float(r)
            # 存储
            name = fr".\result\{data_name}\UDAN_{input_name}"
            img = myNormalization(gen_img) * 255
            img = Image.fromarray(img).convert("L")
            img.save(name)
        i += 1

std_s = np.std(std_all1)
std_p = np.std(std_all2)
std_r = np.std(std_all3)
aver_time = total_time / (len(test_loader_source) * batch_size)
print(f"{len(test_loader_source) * batch_size}张图片，总用时：{total_time},平均用时为:{aver_time}")

print(f'{data_name}: |Epoch %d/%d| |Average SSIM: %f ± %f | |Average RMSE: %f ± %f| |Average PSNR: %f ± %f|'
      % ((epoch + 1), epoch_total, total_s_source / (len(test_loader_source) * batch_size), std_s,
         total_r_source / (len(test_loader_source) * batch_size), std_r,
         total_p_source / (len(test_loader_source) * batch_size), std_p))
print('\n')
with open(r".\test_parameters_compare.txt", "a") as f:
    f.write(f"{data_name}: |Epoch %d / %d | |Average SSIM: %f ± %f | |Average RMSE: %f ± %f| |Average PSNR: %f ± %f|\n"
            % ((epoch + 1), epoch_total, total_s_source / (len(test_loader_source) * batch_size), std_s,
               total_r_source / (len(test_loader_source) * batch_size), std_r,
               total_p_source / (len(test_loader_source) * batch_size), std_p))
