#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

try:
    from .neural_blocks import *
except:
    from neural_blocks.neural_blocks import *



class autoencoder(nn.Module):
    """
    Autoencoder wrapper
    """
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()


    def encode(self, input):
        return self.encoder(input).flatten(-3, -1)


    def decode(self, input):
        return self.decoder(input.view(*input.shape[:-1], 4, 4, 4))



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        channels_description = [4, 64, 128, 256, 512, 4]
        kernels_description = [3, 3, 3, 3, 1]
        pooling_description = [2, 2, 2, 2, 0]
        activation_description = ['lrelu'] * (len(kernels_description))
        self.squeeze = Neural_blocks.generate_conv_stack(channels_description, kernels_description, pooling_description, activation_description)


    def forward(self, input):
        return self.squeeze(input)



class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        _channels = [4, 32, 64, 128, 256]
        _kernels = [4, 4, 4, 4]
        _padding = [1, 1, 1, 1]
        _stride = [2, 2, 2, 2]
        _activation = ['lrelu', 'lrelu', 'lrelu', 'lrelu']
        self.unsqueeze = Neural_blocks.generate_deconv_stack(_channels, _kernels, _padding, _stride, _activation)

        # the final image, 4x64x64
        _channels = [256, 32, 4]
        _kernels = [3, 1]
        _pooling = [0, 0]
        _activation = ['lrelu', 'sigmoid']
        self.regenerate = Neural_blocks.generate_conv_stack(_channels, _kernels, _pooling, _activation)


    def forward(self, input):
        return self.regenerate(self.unsqueeze(input))

