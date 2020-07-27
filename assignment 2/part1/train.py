################################################################################
# MIT License
#
# Copyright (c) 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime


import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# from assignment2.part1.dataset import PalindromeDataset
# from assignment2.part1.vanilla_rnn import VanillaRNN
# from assignment2.part1.lstm import LSTM

import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    final_accuracy = []
    seq_list = []

    for i in range(30):
        input_length = config.input_length + i
        # Initialize the model that we are going to use
        if config.model_type == 'RNN':
            model = VanillaRNN(input_length, config.input_dim, config.num_hidden, config.num_classes, device=device)
        elif config.model_type == 'LSTM':
            model = LSTM(input_length, config.input_dim, config.num_hidden, config.num_classes, device=device)

        # Initialize the dataset and data loader (note the +1)
        dataset = PalindromeDataset(input_length+1)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

        # Setup the loss and optimizer
        optimizer = optim.RMSprop(model.parameters(), config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        accuracies = []
        losses = []

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            # Only for time measurement of step through network
            t1 = time.time()
            optimizer.zero_grad()
            prediction = model(batch_inputs.to(device))
            loss = criterion(prediction, batch_targets.to(device))
            loss.backward()

            ############################################################################
            # QUESTION: what happens here and why?
            # this function causes the gradient not to explode
            ############################################################################
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            ############################################################################

            optimizer.step()

            _, predicted = torch.max(prediction, 1)
            accuracy = (predicted == batch_targets.to(device)).sum().item() / len(predicted)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            accuracies.append(accuracy*100)
            losses.append(loss)

            if step % 10 == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))
            if step % 100 == 0:
                # go on with training if the model predicted with 100% accuracy for the last 100 steps
                accuracy_average = sum(accuracies[-100:])/100
                if accuracy_average == 100:
                    break

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

        final_accuracy.append(accuracy_average)
        seq_list.append(input_length)

        print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    # parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
