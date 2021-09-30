#!/usr/bin/env python3

import torch
import torch.nn as nn

try:
    from .neural_blocks import *
except:
    from neural_blocks import *



class condPixelCNN(nn.Module):
    """
    Conditional PixelCNN using gated convolutional layers
    """
    def __init__(self, num_embedding, masks_description, kernel_size):
        super().__init__()

        # conditional pixelcnn
        self.condPixelCNN = Neural_blocks.generate_cond_gated_masked_conv_stack( \
                                                                                channels_description=num_embedding,
                                                                                masks_description=masks_description,
                                                                                kernels_description=kernel_size
                                                                                )

        # head of conditional pixelcnn
        self.head = nn.Conv2d(num_embedding, num_embedding, 1, bias=False)


    def forward(self, encodings, context):
        # check if there is a sequence dimension
        seq_bat = encodings.shape[0]
        if len(encodings.shape) == 5:
            seq_bat = context.shape[:2]
            encodings = encodings.flatten(0, 1)
            context = context.flatten(0, 1)

        # compute prior pixel distribution
        encodings_prior = encodings
        for layer in self.condPixelCNN:
            encodings_prior = layer(encodings_prior, context)

        # 1x1 conv to exit from gated realm, this layer is very important
        encodings_prior = self.head(encodings_prior)

        # convert to multinoulli distribution
        encodings_prior = torch.softmax(encodings_prior, 1)

        # restore sequence dimension if there ever was any
        encodings_prior = encodings_prior.view(*seq_bat, *encodings_prior.shape[1:])

        return encodings_prior


    def sample(self, context):
        # prepare the canvas
        canvas = torch.zeros_like(context)

        for i in range(canvas.shape[-1]):
            for j in range(canvas.shape[-2]):
                # push canvas through pixelcnn and condition on context
                out = self.forward(canvas, context)
                # get a set of probability distributions at the current pixel location
                probs = out[..., i, j]
                # sample the index, sampling from distribution doesn't really work
                # index = dist.Categorical(probs=probs).sample()
                index = torch.argmax(probs, dim=-1)
                # fill the indices sampled with 1 and repeat
                canvas[..., i, j].scatter_(-1, index.unsqueeze(-1), 1)
                # straight through gradients
                canvas[..., i, j] = canvas[..., i, j] - probs.detach() + probs

        return canvas
