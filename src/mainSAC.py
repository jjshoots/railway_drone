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

from ai_lib.replay_buffer import *
from ai_lib.normal_inverse_gamma import *
from ai_lib.UASAC import UASAC


def train(set):
    net, net_helper, net_optim, net_optim_helper, net_sched = setup_nets(set)
    envs = setup_envs(set)
    memory = ReplayBuffer(set.buffer_size)

    for epoch in range(set.start_epoch, set.epochs):
        # gather the data
        net.eval()
        rewards_tracker = []
        entropy_tracker = []

        states = np.zeros((set.num_envs, 3, *envs[0].env.drone.frame_size))
        auxiliary = np.zeros((set.num_envs, 2))
        next_states = np.zeros((set.num_envs, 3, *envs[0].env.drone.frame_size))
        next_auxiliary = np.zeros((set.num_envs, 2))
        actions = np.zeros((set.num_envs, 2))
        next_actions = np.zeros((set.num_envs, 2))
        rewards = np.zeros((set.num_envs, 1))
        dones = np.zeros((set.num_envs, 1))
        labels = np.zeros((set.num_envs, 2))
        next_labels = np.zeros((set.num_envs, 2))
        entropy = np.zeros((set.num_envs, 1))

        for _ in range(int(set.transitions_per_epoch / set.num_envs)):
            net.zero_grad()

            # get the initial state and action
            for i, env in enumerate(envs):
                obs, aux, _, _, lbl = env.get_state()
                states[i] = (obs[..., :-1].transpose(2, 0, 1) - 127.5) / 255.
                auxiliary[i] = aux
                labels[i] = lbl

            output = net.backbone(gpuize(states, set.device), gpuize(auxiliary, set.device))
            output = net.actor(output)
            o1, ent, _ = net.actor.sample(*output)
            actions = cpuize(o1) if epoch > set.pretrain_epochs else labels
            entropy = cpuize(ent)

            # get the next state, next action, and other stuff
            for i, env in enumerate(envs):
                obs, aux, rew, dne, lbl = env.step(actions[i])
                next_states[i] = (obs[..., :-1].transpose(2, 0, 1) - 127.5) / 255.
                next_auxiliary[i] = aux
                rewards[i] = rew
                dones[i] = dne
                next_labels[i] = lbl

                if dne:
                    env.reset()

            output = net.backbone(gpuize(states, set.device), gpuize(auxiliary, set.device))
            output = net.actor(output)
            o1, _, _ = net.actor.sample(*output)
            next_actions = cpuize(o1) if epoch > set.pretrain_epochs else next_labels

            # store stuff in mem
            for stuff in zip(states, auxiliary,
                             next_states, next_auxiliary,
                             actions, next_actions,
                             rewards, dones, labels):
                memory.push(stuff)

            # log progress
            rewards_tracker.append(np.mean(rewards if epoch > set.pretrain_epochs else -10))
            entropy_tracker.append(np.mean(entropy))

        # for logging
        rewards_tracker = -np.mean(np.array(rewards_tracker))
        entropy_tracker = -np.mean(np.array(entropy_tracker))

        # train on data
        net.train()
        dataloader = torch.utils.data.DataLoader(memory, batch_size=set.batch_size, shuffle=True, drop_last=False)

        for i in range(set.repeats_per_buffer):
            for j, stuff in enumerate(dataloader):
                net.zero_grad()

                batch = int(set.buffer_size / set.batch_size) * i + j
                states = gpuize(stuff[0], set.device)
                auxiliary = gpuize(stuff[1], set.device)
                next_states = gpuize(stuff[2], set.device)
                next_auxiliary = gpuize(stuff[3], set.device)
                actions = gpuize(stuff[4], set.device)
                next_actions = gpuize(stuff[5], set.device)
                rewards = gpuize(stuff[6], set.device)
                dones = gpuize(stuff[7], set.device)
                labels = gpuize(stuff[8], set.device)

                q_loss, reg_loss = net.calc_critic_loss(states, auxiliary, next_states, next_auxiliary, actions, next_actions, rewards, dones)
                rnf_loss, sup_loss, sup_scale = net.calc_actor_loss(states, auxiliary, dones, labels)
                ent_loss = net.calc_alpha_loss(states, auxiliary)

                sup_scale = sup_scale if epoch > set.pretrain_epochs else torch.tensor(1.)
                total_loss = q_loss \
                    + (set.reg_lambda * (sup_loss / reg_loss).mean().detach() * reg_loss) \
                    + ((1. - sup_scale) * rnf_loss).mean() \
                    + (sup_scale * sup_loss).mean() \
                    + ent_loss

                total_loss.backward()
                net_optim.step()
                net_sched.step()
                net.update_q_target()

                # detect whether we need to save the weights file and record the losses
                net_weights = net_helper.training_checkpoint(loss=rewards_tracker, batch=batch, epoch=epoch)
                net_optim_weights = net_optim_helper.training_checkpoint(loss=rewards_tracker, batch=batch, epoch=epoch)
                if net_weights != -1: torch.save(net.state_dict(), net_weights)
                if net_optim_weights != -1: torch.save({ \
                                                       'net_optimizer': net_optim.state_dict(),
                                                       'net_scheduler': net_sched.state_dict(),
                                                       'lowest_running_loss': net_optim_helper.lowest_running_loss,
                                                       'epoch': epoch
                                                       },
                                                      net_optim_weights)

                # wandb
                metrics = { \
                            "epoch": epoch, \
                            "mean_reward": rewards_tracker if epoch > set.pretrain_epochs else 0., \
                            "mean_entropy": entropy_tracker, \
                            "sup_scale": sup_scale.mean().item(), \
                            "log_alpha": net.log_alpha.item(), \
                           } \

                if set.wandb:
                    wandb.log(metrics)


def display(set):

    set.num_envs = 1
    set.max_steps = math.inf

    envs = setup_envs(set)
    net, _, _, _, _ = setup_nets(set)
    net.eval()

    target = np.zeros((set.num_envs, 2))
    uncertainty = np.zeros((set.num_envs, 1))
    state = [None] * set.num_envs
    auxiliary = np.zeros((set.num_envs, 2))
    labels = np.zeros((set.num_envs, 2))

    cv2.namedWindow('display', cv2.WINDOW_NORMAL)

    while True:
        for i, env in enumerate(envs):
            obs, aux, _, dne, lbl = env.step(target[i])
            state[i] = obs
            auxiliary[i] = aux
            labels[i] = lbl

            obs = gpuize((obs[..., :-1].transpose(2, 0, 1) - 127.5) / 255., set.device).unsqueeze(0)
            aux = gpuize(aux, set.device).unsqueeze(0)

            if dne:
                env.reset()

            if True:
                output = net.backbone(obs, aux)
                output = net.actor(output)
                target[i] = cpuize(output[0])
            else:
                target[i] = labels

        img = np.concatenate(state, axis=1)

        cv2.imshow('display', img)
        cv2.waitKey(1)


def setup_envs(set):
    envs = \
    [
        Environment(
            rails_dir='models/rails/',
            drone_dir='models/vehicles/',
            plants_dir='models/plants/',
            tex_dir='models/textures/',
            num_envs=set.num_envs,
            max_steps=set.max_steps
            )
        for _ in range(set.num_envs)
    ]

    return envs


def setup_nets(set):
    net_helper = Logger(mark_number=set.tracknet_number,
                         version_number=set.tracknet_version,
                         weights_location=set.weights_directory,
                         epoch_interval=set.epoch_interval,
                         batch_interval=set.batch_interval,
                         )
    net_optim_helper = Logger(mark_number=0,
                               version_number=set.tracknet_version,
                               weights_location=set.optim_weights_directory,
                               epoch_interval=set.epoch_interval,
                               batch_interval=set.batch_interval,
                               increment=False,
                               )

    # set up networks and optimizers
    net = UASAC(num_actions=2, target_entropy=-1.5).to(set.device)
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
