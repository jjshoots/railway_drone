import os
import argparse
from sys import exit
from signal import signal, SIGINT

import cv2
import yaml
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

from utility.shebangs import *

from env.environment import *

from ai_lib.replay_buffer_simple import *


def train(set):
    envs = setup_envs(set)
    memory = ReplayBufferSimple(set.buffer_size)

    action = np.zeros((set.num_envs, 2))

    for epoch in range(set.epochs):
        # gather the data
        memory.counter = 0
        while not memory.is_full():
            for i, env in enumerate(envs):
                obs, _, done, target = env.step(action[i])

                # ommit saving alpha channel of image and flip to pytorch aligned axis
                memory.push(obs[..., :-1].transpose(2, 0, 1), target)
                action[i] = target

                if done:
                    env.reset()

        dataloader = torch.utils.data.DataLoader(memory, batch_size=set.batch_size, shuffle=True, drop_last=False)

        # train
        for _, stuff in enumerate(dataloader):
            obs = stuff[0].to(set.device)
            label = stuff[1].to(set.device)

            print(obs.shape)
            print(label.shape)
            exit()


def display(set):
    envs = setup_envs(set)

    track_state = np.zeros((set.num_envs, 2))
    stack_obs = [None] * set.num_envs

    cv2.namedWindow('display', cv2.WINDOW_NORMAL)

    while True:
        for i, env in enumerate(envs):
            obs, rew, done, target = env.step(track_state[i])

            if done:
                env.reset()

            stack_obs[i] = obs
            track_state[i] = target

        img = np.concatenate(stack_obs, axis=1)

        cv2.imshow('display', img)
        cv2.waitKey(1)


def setup_envs(set):
    envs = \
    [
        Environment(
            rails_dir='models/rails/',
            drone_dir='models/vehicles/',
            tex_dir='models/textures/',
            num_envs=set.num_envs,
            max_steps=set.max_steps
            )
        for _ in range(set.num_envs)
    ]

    return envs


if __name__ == '__main__':
    signal(SIGINT, shutdown_handler)
    set, args = parse_set_args()

    """ SCRIPTS HERE """

    if args.display:
        display(set)
    elif args.train:
        train(set)
    else:
        print('Guess this is life now.')

    """ SCRIPTS END """

    if args.shutdown:
        pass
