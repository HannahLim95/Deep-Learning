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

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.weighthx = nn.Parameter(torch.empty(num_hidden, input_dim, device=device), requires_grad=True)
        self.weighthh = nn.Parameter(torch.empty(num_hidden, num_hidden, device=device), requires_grad=True)
        self.bh = nn.Parameter(torch.empty(num_hidden,1, device = device), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weighthx)
        torch.nn.init.kaiming_uniform_(self.weighthh)
        torch.nn.init.kaiming_uniform_(self.bh)

        self.weightph = nn.Parameter(torch.empty(num_classes, num_hidden, device=device), requires_grad=True)
        self.bp = nn.Parameter(torch.empty(num_classes,1, device=device), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weightph)
        torch.nn.init.kaiming_uniform_(self.bp)

        self.seq_length = seq_length
        self.device = device
        self.num_hidden = num_hidden
        self.input_dim = input_dim



    def forward(self, x):
        h = torch.zeros(self.num_hidden, x.shape[0], device=self.device)
        self.h_list = []

        for i in range(self.seq_length):
            h = torch.tanh(self.weighthx @ x[:,i].view(-1,1).t() + self.weighthh @ h + self.bh)
            h_t = h
            h_t.requires_grad_()
            h_t.retain_grad()
            self.h_list.append(h_t)

        output = self.weightph @ h + self.bp
        return torch.transpose(output, 0, 1)