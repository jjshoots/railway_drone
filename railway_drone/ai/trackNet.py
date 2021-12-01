import torch
import torch.nn as nn
import torch.nn.functional as F

from railway_drone.ai.neural_blocks import *


class TrackNet(nn.Module):
    """
    Network that outputs the track position using evidential DL
    """
    def __init__(self):
        super().__init__()

        _channels_description = [3, 32, 32, 64, 64, 128, 128, 256, 8]
        _kernels_description = [3] * (len(_channels_description) - 1)
        _pooling_description = [2] * (len(_channels_description) - 2) + [0]
        _activation_description = ['lrelu'] * (len(_channels_description) - 2) + ['identity']
        self.net = Neural_blocks.generate_conv_stack(_channels_description, _kernels_description, _pooling_description, _activation_description)


    def forward(self, input):
        input = self.net(input).reshape(-1, 4, 2).permute(1, 0, 2)

        mu, lognu, logalpha, logbeta = torch.split(input, 1, dim=0)

        nu = F.softplus(lognu) + 1e-6
        alpha = F.softplus(logalpha) + 1. + 1e-6
        beta = F.softplus(logbeta) + 1e-6

        return torch.cat([mu, nu, alpha, beta], dim=0)
