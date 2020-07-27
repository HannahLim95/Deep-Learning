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

import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device = device
        self.dropout = 0

        self.lstm = nn.LSTM(vocabulary_size, lstm_num_hidden, lstm_num_layers, dropout=self.dropout)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.embedding = nn.Embedding(batch_size * seq_length, vocabulary_size)


    def forward(self, x, h, c):
        x = self.embedding(x)
        x = x.permute(1,0,2)
        output, (h,c) = self.lstm(x, (h, c))
        output = self.linear(output)
        return output, (h,c)