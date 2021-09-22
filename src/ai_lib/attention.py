#!/usr/bin/env python

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .neural_blocks import *
except:
    from neural_blocks import *


class AttentionHead(nn.Module):
    def __init__(self, latent_dim, head_dim):
        super().__init__()

        self.q = nn.Linear(latent_dim, head_dim)
        self.k = nn.Linear(latent_dim, head_dim)
        self.v = nn.Linear(latent_dim, head_dim)


    def scaled_dot_product_attention(self, query, key, value):
        temp = query.matmul(key.transpose(1, 2))
        scale = query.shape[-1] ** 0.5
        softmax = F.softmax(temp / scale, dim=-1)
        return softmax.matmul(value)


    def forward(self, query, key, value):
        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        return self.scaled_dot_product_attention(query, key, value)



class TransformerLayer(nn.Module):
    def __init__(self, latent_dim, head_dim, num_head, ff_depth=1, ff_hidden_dim=128):
        super().__init__()

        # multihead attention
        self.heads = nn.ModuleList([AttentionHead(latent_dim, head_dim) for _ in range(num_head)])
        self.linear = nn.Linear(num_head * head_dim, latent_dim)
        self.norm1 = nn.LayerNorm(latent_dim)

        # feedforward and add and norm
        channels = [latent_dim] + [ff_hidden_dim] * ff_depth + [latent_dim]
        activation = ['lrelu'] * (len(channels) - 1)
        self.feedforward = Neural_blocks.generate_linear_stack(channels, activation, batch_norm=False)
        self.norm2 = nn.LayerNorm(latent_dim)


    def forward(self, query, key, value):
        output = torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        output = self.linear(output) + query
        output = self.norm1(output)

        output = self.feedforward(output) + output
        output = self.norm2(output)

        return output



def create_positional_encoding(latent_size, latent_dim):
    """
    Creates a positional encoding
    """
    # determine the sizes we need
    num_slices = int(math.log(latent_dim, 2))
    assert (math.log(latent_size, 2) % 1 == 0), 'Width of latent space must be power of 2'

    # generate a blank block to fill up
    shape = [1, latent_size, int(num_slices)]
    block = torch.zeros(shape)

    for i in range(int(num_slices)):
        for j in range(latent_size):
            state = (j % (latent_size / (2 ** i)) < latent_size / (2 ** (i+1))) * 1.
            block[:, j, i] = state

    # repeat a number of times as necessary to form encoding
    num_repeat = math.ceil(latent_dim / num_slices)
    return torch.cat([block] * num_repeat, dim=-1)[:, :, :latent_dim]
