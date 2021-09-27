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

import torch
import torch.nn.functional as F

from utility.shebangs import *

from env.environment import *

from ai_lib.replay_buffer_simple import *
from ai_lib.normal_inverse_gamma import *
from ai_lib.trackNet import *


def train(set):
    net, net_helper, net_optim, net_optim_helper, net_sched = setup_nets(set)
    envs = setup_envs(set)
    memory = ReplayBufferSimple(set.buffer_size)

    stacked_obs = torch.zeros(set.num_envs, 3, *envs[0].env.drone.frame_size).to(set.device)
    dones = np.zeros((set.num_envs, 1))
    targets = np.zeros((set.num_envs, 2))
    actions = np.zeros((set.num_envs, 2))

    for epoch in range(set.start_epoch, set.epochs):
        # gather the data
        memory.counter = 0
        while not memory.is_full():
            for i, env in enumerate(envs):
                obs, _, dne, tgt = env.step(actions[i])

                if not dne:
                    # ommit saving alpha channel of image and flip to pytorch aligned axis
                    obs = (obs[..., :-1].transpose(2, 0, 1) - 127.5) / 255.
                    memory.push(obs, tgt)

                    stacked_obs[i] = torch.tensor(obs).float()
                    dones[i] = dne
                    targets[i] = tgt
                else:
                    env.reset()

            if epoch < 5:
                actions = targets * dones
            else:
                actions = net.forward(stacked_obs)[0].squeeze(0).detach().cpu().numpy()
                actions = actions * dones

        dataloader = torch.utils.data.DataLoader(memory, batch_size=set.batch_size, shuffle=True, drop_last=False)

        # train on data
        for i in range(set.repeats_per_buffer):
            for j, stuff in enumerate(dataloader):
                batch = int(set.buffer_size / set.batch_size) * i + j
                net.zero_grad()

                obs = stuff[0].to(set.device)
                label = stuff[1].to(set.device)
                output = net.forward(obs)

                pred_loss =NIG_NLL(label, *output)
                evid_loss = set.reg_lambda * NIG_reg(label, *output)
                total_loss = pred_loss + evid_loss

                # pred_loss = F.mse_loss(output[0], label)
                # total_loss = pred_loss

                total_loss.backward()
                net_optim.step()
                net_sched.step()

                # detect whether we need to save the weights file and record the losses
                pred_loss = F.mse_loss(output[0], label)
                net_weights = net_helper.training_checkpoint(loss=pred_loss.data, batch=batch, epoch=epoch)
                net_optim_weights = net_optim_helper.training_checkpoint(loss=pred_loss.data, batch=batch, epoch=epoch)
                if net_weights != -1: torch.save(net.state_dict(), net_weights)
                if net_optim_weights != -1: torch.save({ \
                                                       'net_optimizer': net_optim.state_dict(),
                                                       'net_scheduler': net_sched.state_dict(),
                                                       'lowest_running_loss': net_optim_helper.lowest_running_loss,
                                                       'epoch': epoch
                                                       },
                                                      net_optim_weights)


def display(set):
    set.num_envs = 1
    set.max_steps = math.inf

    envs = setup_envs(set)
    net, _, _, _, _ = setup_nets(set)
    net.eval()

    target = np.zeros((set.num_envs, 2))
    uncertainty = np.zeros((set.num_envs, 1))
    stack_obs = [None] * set.num_envs

    cv2.namedWindow('display', cv2.WINDOW_NORMAL)

    while True:
        for i, env in enumerate(envs):
            obs, _, done, info = env.step(target[i])

            stack_obs[i] = obs

            if True:
                obs = (obs[..., :-1].transpose(2, 0, 1) - 127.5) / 255.
                obs = torch.tensor(obs).unsqueeze(0).to(set.device).type(torch.float32)

                output = net.forward(obs)
                target[i] = output[0].squeeze(0).detach().cpu().numpy()
                uncertainty[i] = torch.mean(NIG_uncertainty(output[-2], output[-1])).squeeze(0).detach().cpu().numpy()
            else:
                target[i] = info

            if done:
                env.reset()

        img = np.concatenate(stack_obs, axis=1)

        print(uncertainty)

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


def setup_nets(set):
    net_helper = helpers(mark_number=set.tracknet_number,
                         version_number=set.tracknet_version,
                         weights_location=set.weights_directory,
                         epoch_interval=set.epoch_interval,
                         batch_interval=set.batch_interval,
                         )
    net_optim_helper = helpers(mark_number=0,
                               version_number=set.tracknet_version,
                               weights_location=set.optim_weights_directory,
                               epoch_interval=set.epoch_interval,
                               batch_interval=set.batch_interval,
                               increment=False,
                               )

    # set up networks and optimizers
    net = TrackNet().to(set.device)
    net_optim = optim.AdamW(net.parameters(), lr=set.starting_LR, amsgrad=True)
    net_sched = optim.lr_scheduler.StepLR(net_optim, step_size=set.step_sched_num, gamma=set.scheduler_gamma)

    # get latest weight files
    net_weights = net_helper.get_weight_file()
    if net_weights != -1: net.load_state_dict(torch.load(net_weights))

    # get latest optimizer states
    net_optimizer_weights = net_optim_helper.get_weight_file()
    if net_optimizer_weights != -1:
        checkpoint = torch.load(net_optimizer_weights)
        net_optim.load_state_dict(checkpoint['net_optimizer'])
        net_sched.load_state_dict(checkpoint['net_scheduler'])
        net_helper.lowest_running_loss = checkpoint['lowest_running_loss']
        net_optim_helper.lowest_running_loss = checkpoint['lowest_running_loss']
        set.start_epoch = checkpoint['epoch']
        print(f'Lowest Running Loss for trackNet: {net_helper.lowest_running_loss} @ epoch {set.start_epoch}')

    return \
        net, net_helper, net_optim, net_optim_helper, net_sched


if __name__ == '__main__':
    signal(SIGINT, shutdown_handler)
    set, args = parse_set_args()
    torch.autograd.set_detect_anomaly(True)

    """ SCRIPTS HERE """

    if args.display:
        display(set)
    elif args.train:
        train(set)
    else:
        print('Guess this is life now.')

    """ SCRIPTS END """

    if args.shutdown:
        os.system('poweroff')
