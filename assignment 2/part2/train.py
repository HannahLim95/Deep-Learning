# MIT License
#
# Copyright (c) 2019 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

# from assignment2.part2.dataset import TextDataset
# from assignment2.part2.model import TextGenerationModel

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import string
from random import randrange

################################################################################

def sample(a, temperature=1.0):
    prediction_vector = F.softmax(a / temperature, dim=1)
    prediction_vector = prediction_vector.squeeze()
    x_index_t = torch.multinomial(prediction_vector, 1).item()
    return x_index_t

def train(config):
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    vocabulary_size = dataset.vocab_size

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, vocabulary_size, config.lstm_num_hidden, config.lstm_num_layers, device = device)

    # Setup the loss and optimizer
    optimizer = optim.RMSprop(model.parameters(), config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    accuracies = []
    losses = []
    h0 = torch.zeros(config.lstm_num_layers, config.batch_size, config.lstm_num_hidden)
    c0 = torch.zeros(config.lstm_num_layers, config.batch_size, config.lstm_num_hidden)
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Only for time measurement of step through network
        t1 = time.time()

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        model.train()

        optimizer.zero_grad()

        prediction, _ = model(batch_inputs, h0, c0)

        loss = criterion(prediction.permute(1, 2, 0), batch_targets)

        loss.backward()

        #######################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        #######################################################

        optimizer.step()

        _, prediction = prediction.max(-1)

        accuracy = (prediction.t() == batch_targets).sum().item() / (prediction.shape[0] * prediction.shape[1])

        accuracies.append(accuracy*100)
        losses.append(loss.item())

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04f}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step % config.sample_every == 0:
            # temperature from [0.5, 1.0, 2.0]
            temp = 0.5

            model.eval()

            h1 = torch.zeros(config.lstm_num_layers, 1, config.lstm_num_hidden)
            c1 = torch.zeros(config.lstm_num_layers, 1, config.lstm_num_hidden)

            # set first character to be a random symbol from the vocabulary
            symbol = torch.randint(low=0, high=dataset.vocab_size, size=(1,1)).long().to(device)

            # uppercase alphabet
            # alphabet = list(string.ascii_uppercase)

            # lowercase alphabet
            # alphabet = list(string.ascii_lowercase)

            # initializing with a random upper- or lowercase letter from the alphabet
            # symbol = torch.tensor([dataset.convert_to_idx(alphabet[randrange(26)])])

            # first character to be 'S'
            # symbol = torch.tensor([dataset.convert_to_idx('S')])

            generated_text = []
            generated_text.append(symbol.item())

            generated_seq_length = 60
            for i in range(generated_seq_length):
                pred_symbol, (h1,c1) = model(symbol, h1, c1)

                # without using temperature
                _, prediction_symbol = pred_symbol.max(-1)
                symbol = prediction_symbol

                # using temperature function
                # symbol = torch.tensor([[sample(pred_symbol, temperature=temp)]])

                generated_text.append(symbol.item())

            # print(dataset.convert_to_string(generated_text))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

    ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)


