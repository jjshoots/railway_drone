import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_lib.neural_blocks import *


class Backbone(nn.Module):
    """
    Backbone for TrackNetUASAC
    """
    def __init__(self):
        super().__init__()

        _channels_description = [3, 32, 32, 64, 64, 128, 128, 256]
        _kernels_description = [3] * (len(_channels_description) - 1)
        _pooling_description = [2] * (len(_channels_description) - 1)
        _activation_description = ['lrelu'] * len(_kernels_description)
        self.vision = Neural_blocks.generate_conv_stack(_channels_description, _kernels_description, _pooling_description, _activation_description)

        _features_description = [2, 128, 256]
        _activation_description = ['lrelu'] * (len(_features_description) - 1)
        self.motion = Neural_blocks.generate_linear_stack(_features_description, _activation_description)


    def forward(self, state, auxiliary):
        return torch.flatten(self.vision(state), -3, -1) + self.motion(auxiliary)


class ActorHead(nn.Module):
    """
    Actor Head for TrackNet
    """
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

        _features_description = [256, num_actions * 4]
        _activation_description = ['identity']
        self.net = Neural_blocks.generate_linear_stack(_features_description, _activation_description)


    def forward(self, states):
        states = self.net(states).reshape(-1, 4, self.num_actions).permute(1, 0, 2)

        mu, lognu, logalpha, logbeta = torch.split(states, 1, dim=0)

        nu = F.softplus(lognu) + 1e-6
        alpha = F.softplus(logalpha) + 1. + 1e-6
        beta = F.softplus(logbeta) + 1e-6

        return torch.cat([mu, nu, alpha, beta], dim=0)


class CriticHead(nn.Module):
    """
    Critic Head for TrackNet
    """
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

        _features_description = [256, 256, 256, 256]
        _activation_description = ['lrelu'] * (len(_features_description) - 1)
        self.first = Neural_blocks.generate_linear_stack(_features_description, _activation_description)

        _features_description = [num_actions, 256]
        _activation_description = ['lrelu']
        self.second = Neural_blocks.generate_linear_stack(_features_description, _activation_description)

        _features_description = [256, 1]
        _activation_description = ['identity']
        self.third = Neural_blocks.generate_linear_stack(_features_description, _activation_description, batch_norm=False)


    def forward(self, states, actions):
        return self.third(self.first(states) + self.second(actions))
