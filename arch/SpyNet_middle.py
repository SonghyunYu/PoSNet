#!/usr/bin/env python

import torch
from matplotlib import pyplot as plt
import numpy as np
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import torch.nn as nn
import scipy
from arch.color_DnCNN import DnCNN


arguments_strModel = 'sintel-final'
Backward_tensorGrid = {}


def Backward(tensorInput, tensorFlow, last_flag=False, factor=0.5):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1,
                                                                                                                  tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1,
                                                                                                                tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()
    # en
    tensorFlow = torch.cat(
    [tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)


    if last_flag:
        return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow*factor).permute(0, 2, 3, 1),
                                               mode='bilinear', padding_mode='border')
    else:
        return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1),
                                             mode='bilinear', padding_mode='border')



class SpyNet_middle(torch.nn.Module):
    def __init__(self):
        super(SpyNet_middle, self).__init__()


        class BGR_to_RGB(torch.nn.Module):
            def __init__(self):
                super(BGR_to_RGB, self).__init__()

            def forward(self, tensorInput):  # BGR to RGB
                tensorBlue = tensorInput[:, 0:1, :, :]
                tensorGreen = tensorInput[:, 1:2, :, :]
                tensorRed = tensorInput[:, 2:3, :, :]

                return torch.cat((tensorRed, tensorGreen, tensorBlue), 1)

        class Normalize(torch.nn.Module):
            def __init__(self):
                super(Normalize, self).__init__()

            def forward(self, tensorInput):  #  imagenet mean, std로 normalize시켜줌.
                tensorRed = (tensorInput[:, 0:1, :, :] - 0.406) / 0.225
                tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
                tensorBlue = (tensorInput[:, 2:3, :, :] - 0.485) / 0.229

                return torch.cat((tensorRed, tensorGreen, tensorBlue), 1)
        # end

        # end

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                self.moduleBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

            # end

            def forward(self, tensorInput):
                return self.moduleBasic(tensorInput)
        # end

        # end

        self.bgr_to_rgb = BGR_to_RGB()
        self.normalize = Normalize()
        self.moduleBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(6)])
        self.load_state_dict(torch.load('./arch/SpyNet_weights/network-' + arguments_strModel + '.pytorch'))



    # end

    def forward(self, tensorFirst, tensorSecond, middle = False):

        tensorFirst_ = [self.normalize(tensorFirst)]
        tensorSecond_ = [self.normalize(tensorSecond)]

        for intLevel in range(5):
            if tensorFirst_[0].size(2) > 2 or tensorFirst_[0].size(3) > 2:  # 32에서 2로 바꿈
                tensorFirst_.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst_[0], kernel_size=2, stride=2, count_include_pad=False))  # [1 3 208 512], [1 3 416 1024] 이런식으로 앞에 작은게 append됨.
                tensorSecond_.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond_[0], kernel_size=2, stride=2, count_include_pad=False))
        # end

        tensorFlow = tensorFirst_[0].new_zeros([tensorFirst_[0].size(0), 2, int(math.floor(tensorFirst_[0].size(2) / 2.0)), int(math.floor(tensorFirst_[0].size(3) / 2.0))])

        for intLevel in range(len(tensorFirst_)):
            tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if tensorUpsampled.size(2) != tensorFirst_[intLevel].size(2): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled,
                                                                                                                   pad=[0, 0, 0, 1], mode='replicate')
            if tensorUpsampled.size(3) != tensorFirst_[intLevel].size(3): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled,
                                                                                                                   pad=[0, 1, 0, 0], mode='replicate')

            tensorFlow = self.moduleBasic[intLevel](
                torch.cat([tensorFirst_[intLevel], Backward(tensorInput=tensorSecond_[intLevel], tensorFlow=tensorUpsampled), tensorUpsampled],
                          1)) + tensorUpsampled


        return Backward(tensorSecond, tensorFlow, last_flag=True, factor=0.5), tensorFlow*0.5



# end

# end
