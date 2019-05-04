################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import math

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = num_hidden
        self.num_classes = num_classes

        if 'cuda' in device.lower() and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(device)

        #Matrices multiplied with input
        self.w_gx = nn.Parameter( torch.randn(input_dim,num_hidden,device=self.device))
        self.w_ix = nn.Parameter(torch.randn(input_dim, num_hidden,device=self.device))
        self.w_fx = nn.Parameter(torch.randn(input_dim, num_hidden,device=self.device))
        self.w_ox = nn.Parameter(torch.randn(input_dim, num_hidden,device=self.device))

        #Matrices multiplied with previous hidden state
        self.w_gh = nn.Parameter(torch.rand(num_hidden, num_hidden,device=self.device))
        self.w_ih = nn.Parameter(torch.rand(num_hidden, num_hidden,device=self.device))
        self.w_fh = nn.Parameter(torch.rand(num_hidden, num_hidden,device=self.device))
        self.w_oh = nn.Parameter(torch.rand(num_hidden, num_hidden,device=self.device))

        #Biases
        self.b_g = nn.Parameter(torch.rand(num_hidden,device=self.device))  # ,1))
        self.b_i = nn.Parameter(torch.rand(num_hidden,device=self.device))  # ,1))
        self.b_f = nn.Parameter(torch.rand(num_hidden,device=self.device))  # ,1))
        self.b_o = nn.Parameter(torch.rand(num_hidden,device=self.device))  # ,1))

        #self.b_g = nn.Parameter(torch.zeros(num_hidden))  # ,1))
        #self.b_i = nn.Parameter(torch.zeros(num_hidden))  # ,1))
        #self.b_f = nn.Parameter(torch.zeros(num_hidden))  # ,1))
        #self.b_o = nn.Parameter(torch.zeros(num_hidden))  # ,1))

        #Initial
        self.prev_h = torch.zeros(batch_size,num_hidden,device=self.device)
        self.prev_c = torch.zeros(batch_size, num_hidden,device=self.device)

        #Output mapping
        self.w_ph = nn.Parameter(torch.randn(num_hidden, num_classes,device=self.device))
        self.b_p = nn.Parameter(torch.rand(num_classes,device=self.device))  # ,1))

        #From NLP1: This is PyTorch's default initialization method
        #stdv = 1.0 / math.sqrt(num_hidden) #num_hidden
        #for weight in self.parameters():
        #   weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # Implementation here ...
        #pass

        prev_h = self.prev_h
        prev_c = self.prev_c

        #For every timestep input in x
        for i in range(self.seq_length):
            single_x_batch = x[:,i]
            single_x_batch = single_x_batch.view((single_x_batch.shape[0],1))
            g_inside = single_x_batch @ self.w_gx + prev_h @ self.w_gh + self.b_g
            g = torch.tanh(g_inside)

            i_inside = single_x_batch @ self.w_ix + prev_h @ self.w_ih + self.b_i
            i = torch.sigmoid(i_inside)

            f_inside = single_x_batch @ self.w_fx + prev_h @ self.w_fh + self.b_f
            f = torch.sigmoid(f_inside)

            o_inside = single_x_batch @ self.w_ox + prev_h @ self.w_oh + self.b_o
            o = torch.sigmoid(o_inside)

            c = g * i + prev_c * f
            prev_h = torch.tanh(c) * o

        #Only compute the cross-entropy for the last timestep, so only last output is needed
        p = prev_h @ self.w_ph + self.b_p

        return p