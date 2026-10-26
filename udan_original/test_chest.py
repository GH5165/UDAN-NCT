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

from math import sqrt

import numpy as np

from torch.utils.data import DataLoader

from datasets import *

import scipy.io as sio



from other.tool.myNormalization import myNormalization





def test_chest(epoch, modelroot, batch_size, epoch_total):

    total_s_source = 0  # ssim

    total_p_source = 0  # psnr

    total_r_source = 0  # rmse



    # model_root = 'models'

    # image_root = os.path.join('dataset', dataset_name)



    cuda = True

    cudnn.benchmark = True



    # 结果图 结果数值 数据集 都放统一的地方

    """load data"""

    data_name = "FDK_raw_200Views_noise1e4"

    test_loader_source = DataLoader(

        # SourceImageDataset(labels_root='',
        # SourceImageDataset(labels_root='',
        SourceImageDataset(labels_root='',  #
                           # inputs_root=rf''),
                           inputs_root=rf''),  #
        batch_size=batch_size,

        shuffle=False,

        drop_last=True,

        # sampler=DistributedSampler(ImageDataset)

    )



    # theta0 = np.linspace(0., 180., 512, endpoint=False)



    """ test """



    my_net = torch.load(modelroot)

    my_net.eval()



    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



    len_test_loader_source = len(test_loader_source)

    # test_loader_source_iter = iter(test_loader_source)



    std_all1 = np.zeros(len_test_loader_source * batch_size)

    std_all2 = np.zeros(len_test_loader_source * batch_size)

    std_all3 = np.zeros(len_test_loader_source * batch_size)



    n_total_source = 0

    i = 0

    total_time = 0

    # while i < len_test_loader_source:

    for i, data_source in enumerate(test_loader_source):



        with torch.no_grad():

            # data_source = test_loader_source_iter.next()

            # inputs, labels, LL2, LL3, _, _ = data_source

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



            start = time.process_time()

            netout, _, _ = my_net(inputs, inputs, LL2, LL3)

            end = time.process_time()

            during = end - start

            total_time += during

            # print(during)

            # print(gen_label_sino.size())

            # gen_label_sino = gen_label_sino.squeeze(-3)

            netout = netout.squeeze(1)

            labels = labels.squeeze(1)

            # netout = Variable(torch.squeeze(netout, dim=1).float(), requires_grad=False)



            # print(gen_label_sino.size())

            netout = netout.cpu().numpy()

            labels = labels.cpu().numpy()

            # print(gen_label_sino.size())



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

                # n_total_source += 1

                # 存储

                name = rf".\img_model6\UDAN_{input_name}"

                # name = rf".\{modelroot[:-12]}\UDAN_{input_name}"

                img = myNormalization(gen_img) * 255

                img = Image.fromarray(img).convert("L")

                img.save(name)



            i += 1



    # labels = torch.from_numpy(labels)

    # gen_label_sino = torch.from_numpy(gen_label_sino)

    # inputs = inputs.squeeze(0)

    # inputs = inputs.squeeze(0)

    # print('inputs shape', inputs.shape)



    aver_s = total_s_source / (len(test_loader_source) * batch_size)

    aver_r = total_r_source / (len(test_loader_source) * batch_size)

    aver_p = total_p_source / (len(test_loader_source) * batch_size)

    std_s = np.std(std_all1)

    std_p = np.std(std_all2)

    std_r = np.std(std_all3)

    aver_time = total_time / (len(test_loader_source) * batch_size)

    print(f"{len(test_loader_source) * batch_size}张图片，总用时：{total_time},平均用时为:{aver_time}")



    # sample = torch.cat((labels, inputs, gen_label_sino), -2)

    # save_image(gen_label_sino, '' % (epoch+1), normalize=False)
    print(f'{data_name}: |Epoch %d/%d| |Average SSIM: %f ± %f | |Average RMSE: %f ± %f| |Average PSNR: %f ± %f|'

          % ((epoch + 1), epoch_total, aver_s, std_s, aver_r, std_r, aver_p, std_p))

    print('\n')

    with open(rf".\saved_evaluations\test_parameters_{modelroot[:-12]}.txt", "a") as f:

        f.write(f'{data_name}: |Epoch %d/%d| |Average SSIM: %f ± %f | |Average RMSE: %f ± %f| |Average PSNR: %f ± %f|\n'

                % ((epoch + 1), epoch_total, aver_s, std_s, aver_r, std_r, aver_p, std_p))

    f.close()

    return aver_s, aver_r, aver_p





if __name__ == '__main__':

    # test_chest(epoch=50, modelroot='./saved_models/model_current.pkl', batch_size=1, epoch_total=50)

    # test_chest(237, r'saved_models\sparse_model_best238.pkl', 1, 300)

    test_chest(37, r'saved_models\noise2_model6_best38.pkl', 1, 100)

    pass

