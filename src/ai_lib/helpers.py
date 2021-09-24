#!/usr/bin/env python3
from sys import version
import time
import os
import sys
import numpy as np
from pthflops import count_ops

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF

from PIL import Image
import matplotlib.pyplot as plt
import cv2


class helpers:
    def __init__(self, mark_number, version_number, weights_location, epoch_intermediary=10, iteration_intermediary=3, greater_than=0.0, increment=True):
        # training checkpoint variables
        self.running_loss = 0
        self.last_saved_epoch = 0
        self.lowest_running_loss = 0
        self.increment = increment
        self.epoch_intermediary = epoch_intermediary
        self.iteration_intermediary = iteration_intermediary
        self.greater_than = greater_than

        # weight file variables
        self.directory = os.path.dirname(__file__)
        self.weights_location = os.path.join(self.directory, f'../../{weights_location}')
        self.mark_number = mark_number
        self.version_number = version_number
        self.weights_file = os.path.join(self.weights_location, f'Version{self.version_number}/weights{self.mark_number}.pth')
        self.weights_file_short = weights_location

        # record that we're in a new training session
        if not os.path.isfile(os.path.join(self.weights_location, f'Version{self.version_number}/training_log.txt')):
            print(F"No weights directory for {self.weights_file_short}/Version{self.version_number}, generating new one in 3 seconds.")
            time.sleep(3)
            os.makedirs(os.path.join(self.weights_location, f'Version{self.version_number}'))

        f = open(os.path.join(self.weights_location, f'Version{self.version_number}/training_log.txt'), "a")
        f.write(f'New Training Session, Net Version {self.version_number} \n')
        f.write(f'Epoch, iterations, Running Loss, Lowest Running Loss, Mark Number \n')
        f.close()



    # simple call to reset internal running loss
    def reset_running_loss(self):
        self.running_loss = 0.


    # call inside training loop
    # helps to display training progress and save any improvement
    def training_checkpoint(self, loss, iterations, epoch, readout=True):

        # if we don't need to increment the mark number
        if not self.increment:
            readout = False

        # reset running loss if we start a new epoch
        if iterations <= 0:
            self.reset_running_loss()

        # add loss to total running loss
        self.running_loss += loss

        if (iterations % self.iteration_intermediary == 0 and iterations != 0) or iterations < 0:
            # at the moment, no way to evaluate the current state of training, so we just record the current running loss
            self.lowest_running_loss = (self.running_loss if (self.lowest_running_loss == 0.0) else self.lowest_running_loss)

            # print status
            if readout:
                print(f'Epoch {epoch}; Batch Number {iterations}; Running Loss {self.running_loss:.5f}; Lowest Running Loss {self.lowest_running_loss:.5f}')

            # only record log files if we're reading out
            if readout:
                # record training log
                f = open(os.path.join(self.weights_location, f'Version{self.version_number}/training_log.txt'), "a")
                f.write(f'{epoch}, {iterations}, {self.running_loss}, {self.lowest_running_loss}, {self.mark_number} \n')
                f.close()

            # save the current weights to an intermediary file if we haven't saved in the last 200 epochs
            # we save it as an intermediary file to prevent messing with the so far best weights file
            if epoch - self.last_saved_epoch >= self.epoch_intermediary and epoch > 0:
                # record the last time we saved a file
                self.last_saved_epoch = epoch

                # save the weights file
                self.weights_file = os.path.join(self.weights_location, f'Version{self.version_number}/weights_intermediary.pth')
                if readout:
                    print(F"Passed {self.epoch_intermediary} epochs without saving so far, saving weights to: /weights_intermediary.pth")
                return self.weights_file

            # save the network if the current running loss is lower than the one we have
            if (self.running_loss + self.greater_than) < self.lowest_running_loss and epoch > 0:
                # record the last time we saved a file
                self.last_saved_epoch = epoch

                # save the net
                self.lowest_running_loss = self.running_loss

                # only increment the mark number if we want to increment
                if self.increment:
                    self.mark_number += 1

                # regenerate the weights_file path
                self.weights_file = os.path.join(self.weights_location, f'Version{self.version_number}/weights{self.mark_number}.pth')

                # reset the running loss for the next n batches
                self.running_loss = 0.

                # print the weight file that we should save to
                if readout:
                    print(F"New lowest point, saving weights to: {self.weights_file_short}/weights{self.mark_number}.pth")
                return self.weights_file

            # reset the running loss for the next n batches
            self.reset_running_loss()

        # return -1 if we are not returning the weights file
        return -1


    def write_auxiliary(self, data: np.ndarray, variable_name: str, precision: str ='%1.3f') -> None:
        assert (len(data.shape) == 1), 'Data must be only 1 dimensional ndarray'
        filename = os.path.join(self.weights_location, f'Version{self.version_number}/{variable_name}.csv')
        with open(filename, 'ab') as f:
            np.savetxt(f, [data], delimiter=',', fmt=precision)


    # retrieves the latest weight file based on mark and version number
    # weight location is location where all weights of all versions are stored
    # version number for new networks, mark number for training
    def get_weight_file(self, latest=True):

        # if we are not incrementing and the file doesn't exist, just exit
        if not self.increment:
            if not os.path.isfile(self.weights_file):
                return -1

        # if we don't need the latest file, get the one specified
        if not latest:
            if os.path.isfile(self.weights_file):
                self.weights_file = os.path.join(self.weights_location, f'Version{self.version_number}/weights{self.mark_number}.pth')
                return self.weights_file

        # while the file exists, try to look for a file one version later
        while os.path.isfile(self.weights_file):
            self.mark_number += 1
            self.weights_file = os.path.join(self.weights_location, f'Version{self.version_number}/weights{self.mark_number}.pth')

        # once the file version doesn't exist, decrement by one and use that file
        self.mark_number -= 1
        self.weights_file = os.path.join(self.weights_location, f'Version{self.version_number}/weights{self.mark_number}.pth')

        # if there's no files, ignore, otherwise, print the file
        if os.path.isfile(self.weights_file):
            print(F"Using weights file: /{self.weights_file_short}/Version{self.version_number}/weights{self.mark_number}.pth")
            return self.weights_file
        else:
            print(F"No weights file found, generating new one during training.")
            return -1


#################################################################################################
#################################################################################################
# STATIC FUNCTIONS
#################################################################################################
#################################################################################################

    @staticmethod
    def get_device():
        # select device
        device = 'cpu'
        if(torch.cuda.is_available()):
            device = torch.device('cuda:0')

        print('-----------------------')
        print('Using Device', device)
        print('-----------------------')

        return device



    # enables peeking of the dataset
    @staticmethod
    def peek_dataset(dataloader):
        dataiter = iter(dataloader)
        user_input = None
        while user_input != 'Y':
            data, label = dataiter.next()

            print(label[0].shape)
            print(data[0].shape)

            # I'm lazy
            j = 0
            figure = plt.figure()

            for i in range(2*4):
                figure.add_subplot(4, 4, (i+1))
                plt.imshow(data[j].squeeze(), cmap='Greys')
                figure.add_subplot(4, 4, (i+1)+(2*4))
                plt.imshow(label[j].squeeze(), cmap='Greys')

                j = j + 1

            plt.show()

            user_input = input('Key in "Y" to end display, enter to continue...')

        exit()



    # converts saliency map to pseudo segmentation
    # expects input of dim 2
    # fastener_area_threshold is minimum area of output object BEFORE scaling to input size
    @staticmethod
    def saliency_to_contour(input, original_image, fastener_area_threshold, input_output_ratio):
        # find contours in the image
        threshold = input.detach().cpu().squeeze().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # draw contours
        contour_image = None
        contour_number = 0
        if original_image != None:
            contour_image = original_image.squeeze().to('cpu').detach().numpy()

        for contour in contours:
            if cv2.contourArea(contour) > fastener_area_threshold:
                contour *= input_output_ratio
                contour_number += 1
                if original_image != None:
                    x,y,w,h = cv2.boundingRect(contour)
                    contour_image = contour_image.astype(np.float32)
                    cv2.rectangle(contour_image,(x,y),(x+w,y+h),1,2)

        # return drawn image
        return contour_image, contour_number



    @staticmethod
    def network_stats(network, input_image):
        total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        count_ops(network, input_image)
        print(f'Total number of Parameters: {total_params}')



    @staticmethod
    def fgsm_attack(data, epsilon=0.1):
        # Create the perturbed data by adjusting each pixel of the input data
        data = data + epsilon * data.grad.data.sign()
        # Adding clipping to maintain [0,1] range
        data = torch.clamp(data, 0, 1)

        return data

