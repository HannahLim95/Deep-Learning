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

    input_length = config.input_length
    # Initialize the models
    model_RNN = VanillaRNN(input_length, config.input_dim, config.num_hidden, config.num_classes, device=device)
    model_LSTM = LSTM(input_length, config.input_dim, config.num_hidden, config.num_classes, device=device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    criterion_RNN = nn.CrossEntropyLoss()
    criterion_LSTM = nn.CrossEntropyLoss()
    gradients_RNN = np.ndarray([config.input_length, 1])
    gradients_LSTM = np.ndarray([config.input_length, 1])

    # print(np.shape(gradients_full)
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Only for time measurement of step through network
        t1 = time.time()

        prediction_RNN = model_RNN(batch_inputs.to(device))
        prediction_LSTM = model_LSTM(batch_inputs.to(device))

        loss_RNN = criterion_RNN(prediction_RNN, batch_targets.to(device))
        loss_LSTM = criterion_LSTM(prediction_LSTM,batch_targets.to(device))

        loss_RNN.backward()
        loss_LSTM.backward()

        magnitude_RNN = []
        for j in model_RNN.h_list:
            magnitude_RNN.append(torch.norm(j.grad).item())

        gradients_RNN = np.append(gradients_RNN, np.array(magnitude_RNN)[:, np.newaxis], axis=1)

        magnitude_LSTM = []
        for j in model_LSTM.h_list:
            magnitude_LSTM.append(torch.norm(j.grad).item())

        gradients_LSTM = np.append(gradients_LSTM, np.array(magnitude_LSTM)[:, np.newaxis], axis=1)

        ############################################################################
        # QUESTION: what happens here and why?
        # this function causes the gradient not to explode, so it will avoid unstable training, oscillations and divergence
        ############################################################################
        torch.nn.utils.clip_grad_norm(model_RNN.parameters(), max_norm=config.max_norm)
        ############################################################################
        _, predicted_RNN = torch.max(prediction_RNN, 1)
        accuracy_RNN = (predicted_RNN == batch_targets.to(device)).sum().item() / len(predicted_RNN)

        _, predicted_LSTM = torch.max(prediction_LSTM, 1)
        accuracy_LSTM = (predicted_LSTM == batch_targets.to(device)).sum().item() / len(predicted_LSTM)


        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy_RNN, loss_RNN
            ))

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy_LSTM, loss_LSTM
            ))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    grad_avg_RNN = np.median(gradients_RNN, axis=1)
    grad_avg_LSTM = np.median(gradients_LSTM, axis=1)

    print(grad_avg_RNN)
    print(grad_avg_LSTM)


################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
