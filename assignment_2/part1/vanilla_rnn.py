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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
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
        print(self.device)

        #Parameters
        self.w_hx = nn.Parameter( torch.randn(input_dim,num_hidden,device=self.device))
        self.w_hh = nn.Parameter( torch.rand(num_hidden,num_hidden,device=self.device))
        self.b_h = nn.Parameter( torch.rand(num_hidden,device=self.device)) #,1))
        self.w_ph = nn.Parameter( torch.randn(num_hidden,num_classes,device=self.device))
        self.b_p = nn.Parameter( torch.rand(num_classes,device=self.device)) #,1))

        #Initial h
        self.prev_h = torch.zeros(batch_size,num_hidden,device=self.device)

        #From NLP1: This is PyTorch's default initialization method
        stdv = 1.0 / math.sqrt(num_hidden)
        for weight in self.parameters():
           weight.data.uniform_(-stdv, stdv)


    def forward(self, x):
        # Implementation here ...
        #pass

        #Using self.prev_h directly resulted into error:
        #RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.
        prev_h = self.prev_h

        #For every timestep input in x
        for i in range(self.seq_length):
            single_x_batch = x[:,i]
            single_x_batch = single_x_batch.view((single_x_batch.shape[0],1))
            inside = single_x_batch @ self.w_hx + prev_h @ self.w_hh + self.b_h
            prev_h = torch.tanh(inside)

        #Only compute the cross-entropy for the last timestep, so only last output is needed
        p = prev_h @ self.w_ph + self.b_p


        return p