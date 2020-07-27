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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()

        self.weightgx = nn.Parameter(torch.empty(num_hidden, input_dim, device = device), requires_grad=True)   #initialiseer kaimin uniform?
        self.weightgh = nn.Parameter(torch.empty(num_hidden, num_hidden, device = device), requires_grad=True)
        self.bg = nn.Parameter(torch.empty(num_hidden, 1, device = device), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weightgx)
        torch.nn.init.kaiming_uniform_(self.weightgh)
        torch.nn.init.kaiming_uniform_(self.bg)

        self.weightix = nn.Parameter(torch.empty(num_hidden, input_dim, device = device), requires_grad=True)
        self.weightih = nn.Parameter(torch.empty(num_hidden, num_hidden, device = device), requires_grad=True)
        self.bi = nn.Parameter(torch.empty(num_hidden, 1, device = device), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weightix)
        torch.nn.init.kaiming_uniform_(self.weightih)
        torch.nn.init.kaiming_uniform_(self.bi)

        self.weightfx = nn.Parameter(torch.empty(num_hidden, input_dim, device = device), requires_grad=True)
        self.weightfh = nn.Parameter(torch.empty(num_hidden, num_hidden, device = device), requires_grad=True)
        self.bf = nn.Parameter(torch.empty(num_hidden, 1, device = device), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weightfx)
        torch.nn.init.kaiming_uniform_(self.weightfh)
        torch.nn.init.kaiming_uniform_(self.bf)

        self.weightox = nn.Parameter(torch.empty(num_hidden, input_dim, device = device), requires_grad=True)
        self.weightoh = nn.Parameter(torch.empty(num_hidden, num_hidden, device = device), requires_grad=True)
        self.bo = nn.Parameter(torch.empty(num_hidden, 1, device = device), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weightox)
        torch.nn.init.kaiming_uniform_(self.weightoh)
        torch.nn.init.kaiming_uniform_(self.bo)

        self.weightph = nn.Parameter(torch.empty(num_classes, num_hidden, device = device), requires_grad=True)
        self.bp = nn.Parameter(torch.empty(num_classes, 1, device = device), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weightph)
        torch.nn.init.kaiming_uniform_(self.bp)

        self.seq_length = seq_length
        self.device = device
        self.num_hidden = num_hidden
        self.input_dim = input_dim

    def forward(self, x):

        ht = torch.zeros(self.num_hidden, x.shape[0], device=self.device)
        ct = torch.zeros(self.num_hidden, x.shape[0], device=self.device)
        self.h_list = []

        for i in range(self.seq_length):
            gt = torch.tanh(self.weightgx @ x[:, i].view(-1,1).t() + self.weightgh @ ht + self.bg)
            it = torch.sigmoid(self.weightix @ x[:, i].view(-1,1).t() + self.weightih @ ht + self.bi)
            ft = torch.sigmoid(self.weightfx @ x[:, i].view(-1,1).t()+ self.weightfh @ ht + self.bf)
            ot = torch.sigmoid(self.weightox @ x[:, i].view(-1,1).t() + self.weightoh @ht + self.bo)
            ct = gt * it + ct * ft
            ht = torch.tanh(ct) * ot
            pt = self.weightph @ ht + self.bp
            h_t = ht
            h_t.requires_grad_()
            h_t.retain_grad()
            self.h_list.append(h_t)

        return torch.transpose(pt, 0, 1)
