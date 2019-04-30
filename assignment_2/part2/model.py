# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.vocab_size = vocabulary_size
        self.lstm_num_hidden =  lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers

        self.embed = nn.Embedding(vocabulary_size, lstm_num_hidden) #Here set the embedding size to be equal to hidden size)
        #With torch.no_grad():
        #self.embed.weight.requires_grad = False #If dont train embeddings

        if 'cuda' in device.lower() and torch.cuda.is_available():
            self.device = torch.device('cuda')
            #print("Cuda")
        else:
            self.device = torch.device('cpu')

        #Since the embedding size is equal to hidden size, that is also the input size
        self.rnn = nn.LSTM(input_size = lstm_num_hidden,hidden_size=lstm_num_hidden,num_layers=2)
        self.h_zero = torch.zeros(2, batch_size, lstm_num_hidden)
        self.c_zero = torch.zeros(2, batch_size, lstm_num_hidden)
        #self.h_zero = torch.randn(2, batch_size, lstm_num_hidden)
        #self.c_zero = torch.randn(2, batch_size, lstm_num_hidden)

        #>>> input = torch.randn(5, 3, 10)   #(seq_len, batch, input_size) #seq_length, batch_size, lstm_num_hidden)
        #>>> h0 = torch.randn(2, 3, 20) (num_layers * num_directions, batch, hidden_size)
        #>>> c0 = torch.randn(2, 3, 20) (num_layers * num_directions, batch, hidden_size)
        #>>> output, (hn, cn) = rnn(input, (h0, c0)) (seq_len, batch, num_directions * hidden_size)

        #Linear output mapping
        self.output_mapping = nn.Linear(lstm_num_hidden, vocabulary_size)
        #Input: (N,∗,in_features)(N, *, \text{in\_features})(N,∗,in_features) where ∗*∗ means any number of additional dimensions
        #Output: (N,∗,out_features)(N, *, \text{out\_features})(N,∗,out_features) where all but the last dimension are the same shape as the input.


    def forward(self, x):
        # Implementation here...
        #pass
        x_batch_size = x.shape[1]
        if x_batch_size == self.batch_size:
            h_zero = self.h_zero
            c_zero = self.c_zero
        else:
            h_zero = torch.zeros(2, x_batch_size, self.lstm_num_hidden)
            c_zero = torch.zeros(2, x_batch_size, self.lstm_num_hidden)

        output, (hn, cn) = self.rnn(x, (h_zero, c_zero))
        #print(output.shape)
        output = self.output_mapping(output) # output.view(len(x), -1)
        #print(output.shape)

        return output
