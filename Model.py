import itertools
from functools import partial
from math import erf as erf_, exp as exp_, sin
from typing import Optional

import math
from torch import Tensor
from typing import List, Optional
from torch.optim.optimizer import Optimizer

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import numpy.typing as npt
import numba.types as nbt
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from numpy import pi, sqrt
from sklearn.model_selection import train_test_split
from torch.nn.parameter import Parameter
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()

        """1D Fourier layer: FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes

        self.scale = 1. / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale
                                    * torch.rand(
            self.in_channels,
            self.out_channels,
            self.modes,
            dtype=torch.cdouble))

        # self.einsum_path_ = None

    def batch_complex_mult(self, input, weights):
        """Multiply the complex weights and input using the following
            tensorial contraction along the in_channel:
            (batch, in_channel, x_i) * (in_channel, out_channel, x_w)
                -> (batch, out_channel, x)

            Note that the x dim is kept, so there is no summation
            going on along it.


            In other words, perform a linear transform
            in the complex plane.
        """

        # if self.einsum_path_ is None:
        #     path = np.einsum_path(
        #         'bix,iox->box',
        #         input,
        #         weights,
        #         optimize='optimal')[0]

        # return np.einsum(
        #     'bix,iox->box',
        #     input,
        #     weights,
        #     optimize=self.einsum_path_)

        return torch.einsum(
            'bix,iox->box',
            input,
            weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier coeff's up to a factor of modulus 1 (phase info)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.batch_complex_mult(
            x_ft[:, :, :self.modes],
            self.weights)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))

        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desired channel dimension
            by self.fc0_dim_lift.
        2. 4 layers of integral operators u' = (W + K)(u) with
            W defined by self.bias_w{k}; K defined by self.conv{k}.
        3. Project from the channel space to some lower dimensional space
            by self.fc1_dim_lower, apply a non-linear activation function,
            and project back to the output space by self.fc2.

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes = modes
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0_dim_lift = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv4 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv5 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv6 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv7 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv8 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv9 = SpectralConv1d(self.width, self.width, self.modes)
        self.bias_w0 = nn.Conv1d(self.width, self.width, 1)
        self.bias_w1 = nn.Conv1d(self.width, self.width, 1)
        self.bias_w2 = nn.Conv1d(self.width, self.width, 1)
        self.bias_w3 = nn.Conv1d(self.width, self.width, 1)
        self.bias_w4 = nn.Conv1d(self.width, self.width, 1)
        self.bias_w5 = nn.Conv1d(self.width, self.width, 1)
        self.bias_w6 = nn.Conv1d(self.width, self.width, 1)
        self.bias_w7 = nn.Conv1d(self.width, self.width, 1)
        self.bias_w8 = nn.Conv1d(self.width, self.width, 1)
        self.bias_w9 = nn.Conv1d(self.width, self.width, 1)

        self.fc1_dim_lower = nn.Linear(self.width, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0_dim_lift(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.bias_w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.bias_w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.bias_w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.bias_w3(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv4(x)
        x2 = self.bias_w4(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv5(x)
        x2 = self.bias_w5(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv6(x)
        x2 = self.bias_w6(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv7(x)
        x2 = self.bias_w7(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv8(x)
        x2 = self.bias_w8(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv9(x)
        x2 = self.bias_w9(x)
        x = x1 + x2

        x = x[..., :-self.padding]  # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1_dim_lower(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(
            np.linspace(0, 1, size_x),
            dtype=torch.float64)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])

        return gridx.to(device)


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)