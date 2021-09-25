import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_lib.neural_blocks import *


class TrackNet(nn.Module):
    """
    Network that outputs the track position using evidential DL
    """
    def __init__(self):
        super().__init__()

        _channels_description = [3, 32, 32, 64, 64, 128, 128, 256, 8]
        _kernels_description = [3] * (len(_channels_description) - 1)
        _pooling_description = [2] * (len(_channels_description) - 2) + [0]
        _activation_description = ['lrelu'] * (len(_kernels_description) - 1) + ['identity']
        self.net = Neural_blocks.generate_conv_stack(_channels_description, _kernels_description, _pooling_description, _activation_description)

        offset = torch.tensor([0., 1e-6, 1+1e-6, 1e-6]).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer('offset', offset, persistent=False)

    def forward(self, input):
        input = self.net(input).reshape(-1, 4, 2).permute(1, 0, 2)

        # avoid in place operation to make torch happy
        output = input.clone()
        output[1:] = F.softplus(input[1:])
        output = output + self.offset

        return output