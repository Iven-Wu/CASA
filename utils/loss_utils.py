import torch
import torch.nn as nn
import os
import pdb
import warnings
warnings.filterwarnings("ignore")
import cv2
import re
import sys
import copy
import random
import numpy as np

import torch.nn.functional as F


seed = 2000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class ARAPLoss(nn.Module):
    def __init__(self, points, faces, average=False):
        super(ARAPLoss, self).__init__()
        self.nv = points.shape[0]
        self.nf = faces.shape[0]
        # faces -= 1
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = 1
        laplacian[faces[:, 1], faces[:, 0]] = 1
        laplacian[faces[:, 1], faces[:, 2]] = 1
        laplacian[faces[:, 2], faces[:, 1]] = 1
        laplacian[faces[:, 2], faces[:, 0]] = 1
        laplacian[faces[:, 0], faces[:, 2]] = 1

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda())

    def forward(self, dx, x):
        # lap: Nv Nv
        # dx: N, Nv, 3
        diffx = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).cuda()
        diffdx = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).cuda()
        for i in range(3):
            dx_sub = self.laplacian.matmul(torch.diag_embed(dx[:, :, i]))  # N, Nv, Nv)
            dx_diff = (dx_sub - dx[:, :, i:i + 1])

            x_sub = self.laplacian.matmul(torch.diag_embed(x[:, :, i]))  # N, Nv, Nv)
            x_diff = (x_sub - x[:, :, i:i + 1])

            diffdx += (dx_diff).pow(2)
            diffx += (x_diff).pow(2)

        diff = (diffx - diffdx).abs()
        diff = torch.stack([diff[i][self.laplacian.bool()].mean() for i in range(x.shape[0])])
        # diff = diff[self.laplacian[None].repeat(x.shape[0],1,1).bool()]
        return diff


class LaplacianLoss(nn.Module):
    def __init__(self, points, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = points.shape[0]
        self.nf = faces.shape[0]
        # faces -= 1
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = 1
        laplacian[faces[:, 1], faces[:, 0]] = 1
        laplacian[faces[:, 1], faces[:, 2]] = 1
        laplacian[faces[:, 2], faces[:, 1]] = 1
        laplacian[faces[:, 2], faces[:, 0]] = 1
        laplacian[faces[:, 0], faces[:, 2]] = 1

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda())

    def forward(self, x):
        # lap: Nv Nv
        # dx: N, Nv, 3

        x_sub = (self.laplacian @ x / (self.laplacian.sum(0)[None, :, None] + 1e-6))
        x_diff = (x_sub - x)
        x_diff = (x_diff).pow(2)
        # diff = diff[self.laplacian[None].repeat(x.shape[0],1,1).bool()]
        return torch.mean(x_diff)

class Preframe_ARAPLoss(nn.Module):
    def __init__(self, points, faces, average=False):
        super(Preframe_ARAPLoss, self).__init__()
        self.nv = points.shape[0]
        self.nf = faces.shape[0]
        # faces -= 1
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = 1
        laplacian[faces[:, 1], faces[:, 0]] = 1
        laplacian[faces[:, 1], faces[:, 2]] = 1
        laplacian[faces[:, 2], faces[:, 1]] = 1
        laplacian[faces[:, 2], faces[:, 0]] = 1
        laplacian[faces[:, 0], faces[:, 2]] = 1

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda())

    def forward(self, dx, x):
        # lap: Nv Nv
        # dx: N, Nv, 3
        diffx = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).cuda()
        diffdx = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).cuda()
        for i in range(3):
            dx_sub = self.laplacian.matmul(torch.diag_embed(dx[:, :, i]))  # N, Nv, Nv)
            dx_diff = (dx_sub - dx[:, :, i:i + 1])

            x_sub = self.laplacian.matmul(torch.diag_embed(x[:, :, i]))  # N, Nv, Nv)
            x_diff = (x_sub - x[:, :, i:i + 1])

            diffdx += (dx_diff).pow(2)
            diffx += (x_diff).pow(2)

        diff = (diffx - diffdx).abs()
        diff = (diff * (self.laplacian.bool()[None])).sum(2)
        # diff = torch.stack([diff[i][self.laplacian.bool()].mean() for i in range(x.shape[0])])
        # diff = diff[self.laplacian[None].repeat(x.shape[0],1,1).bool()]
        return diff

class Preframe_LaplacianLoss(nn.Module):
    def __init__(self, points, faces, average=False):
        super(Preframe_LaplacianLoss, self).__init__()
        self.nv = points.shape[0]
        self.nf = faces.shape[0]
        # faces -= 1
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = 1
        laplacian[faces[:, 1], faces[:, 0]] = 1
        laplacian[faces[:, 1], faces[:, 2]] = 1
        laplacian[faces[:, 2], faces[:, 1]] = 1
        laplacian[faces[:, 2], faces[:, 0]] = 1
        laplacian[faces[:, 0], faces[:, 2]] = 1

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda())

    def forward(self, x):
        # lap: Nv Nv
        # dx: N, Nv, 3

        x_sub = (self.laplacian @ x / (self.laplacian.sum(0)[None, :, None] + 1e-6))
        x_diff = (x_sub - x)
        x_diff = (x_diff).pow(2)
        # diff = diff[self.laplacian[None].repeat(x.shape[0],1,1).bool()]
        return x_diff.mean(2)

class OffsetNet(nn.Module):
    def __init__(self, input_ch=3, out_ch=3, W=256):
        super(OffsetNet, self).__init__()
        self.W = W
        self.input_ch = input_ch
        self.out_ch = out_ch

        self.layer1 = nn.Linear(input_ch, W)
        self.layer2 = nn.Linear(W, W)
        self.layer3 = nn.Linear(W, W)
        self.layer4 = nn.Linear(W, out_ch)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)

        return x
