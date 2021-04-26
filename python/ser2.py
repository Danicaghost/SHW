#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/11 18:04
# @Author : XQP
# @File : ser.py

import torch.nn.functional as F
# from senet.baseline import resnet20
from senet.se_resnet1 import se_resnet50
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob
import scipy.io as sio
import torch.hub
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = se_resnet50(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
            # print(output)
        self.model.avgpool.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer

features_dir = './FISR'
def main():
    model = Net()
    # model.load_state_dict(torch.load("seresnet50-60a8950a85b2b.pkl"))
    model = model.cuda()
    model.eval()

    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG','png']
    features = []
    files_list = []
    imgs_path = open("./FISR.txt").read().splitlines()
    # x = os.walk(data_dir)
    # for path, d, filelist in x:
    #     for filename in filelist:
    #         file_glob = os.path.join(path, filename)
    #         files_list.extend(glob.glob(file_glob))
    #
    # print(files_list)
    for i, img in enumerate(imgs_path):
        print("%d %s" % (i, img))
    print("")
    use_gpu = torch.cuda.is_available()
    # for x_path in files_list:
    #     print("x_path" + x_path)
    #     file_name = x_path.split('/')[-1]
    #     fx_path = os.path.join(features_dir, file_name + '.txt')
    # print(fx_path)
    # extractor(x_path, fx_path, model, use_gpu)

    # def extractor(img_path, saved_path, net, use_gpu):
    for i, im in enumerate(imgs_path):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()]
        )

        img = Image.open(im).convert('RGB')
        img = transform(img)
        print(im)
        print(img.shape)

        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
        print(x.shape)

        if use_gpu:
            x = x.cuda()
            model = model.cuda()
        y = model(x).cpu()
        y = torch.squeeze(y)
        y = y.data.numpy()
        print(y.shape)
        # np.savetxt(saved_path, y, delimiter=',')
        feature = np.reshape(y, [1, -1])
        features.append(feature)
    features = np.array(features)
    dic = {'seresnet50': features}
    sio.savemat(features_dir + '/seresnet5022' + '.mat', dic)

if __name__ == '__main__':

    main()
