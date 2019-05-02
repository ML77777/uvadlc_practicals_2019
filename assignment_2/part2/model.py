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
import random


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

        if 'cuda' in device.lower() and torch.cuda.is_available():
            self.device = torch.device('cuda')
            #print("Cuda")
        else:
            self.device = torch.device('cpu')
        print(self.device)

        self.embed = nn.Embedding(vocabulary_size, lstm_num_hidden) #Here set the embedding size to be equal to hidden size)
        #With torch.no_grad():
        #self.embed.weight.requires_grad = False #If dont train embeddings

        #Since the embedding size is equal to hidden size, that is also the input size
        self.rnn = nn.LSTM(input_size = lstm_num_hidden,hidden_size=lstm_num_hidden,num_layers=2)
        self.h_zero = torch.zeros(self.lstm_num_layers, batch_size, lstm_num_hidden,device = self.device)
        self.c_zero = torch.zeros(self.lstm_num_layers, batch_size, lstm_num_hidden,device = self.device)
        #self.h_zero = torch.randn(2, batch_size, lstm_num_hidden)
        #self.c_zero = torch.randn(2, batch_size, lstm_num_hidden)

        #Linear output mapping
        self.output_mapping = nn.Linear(lstm_num_hidden, vocabulary_size)

    def forward(self, x):
        # Implementation here...
        #pass

        x_batch_size = x.shape[1]
        #Check if the batch size of input is equal to given batch size earlier or is lower as leftovers
        if x_batch_size == self.batch_size:
            h_zero = self.h_zero
            c_zero = self.c_zero
        else:
            h_zero = torch.zeros(self.lstm_num_layers , x_batch_size, self.lstm_num_hidden,device = self.device)
            c_zero = torch.zeros(self.lstm_num_layers , x_batch_size, self.lstm_num_hidden,device = self.device)

        output, (hn, cn) = self.rnn(x, (h_zero, c_zero))
        #Map to the vocabulary classes
        output = self.output_mapping(output)

        return output

    def generate_sentence(self,sentence_length,temperature):

        h_n = torch.zeros(self.lstm_num_layers , 1, self.lstm_num_hidden, device=self.device)
        c_n = torch.zeros(self.lstm_num_layers , 1, self.lstm_num_hidden, device=self.device)

        start_letter = random.randint(0,self.vocab_size-1)
        sentence_id = [start_letter]
        start_letter = torch.tensor(start_letter,device=self.device)
        #print(letter)
        input = self.embed(start_letter)
        #print(input.shape)
        input = input.view(1,1,-1)

        if temperature <= 0:

            for i in range(sentence_length):

                output, (h_n, c_n) = self.rnn(input, (h_n, c_n))
                output = self.output_mapping(output)
                #print(output.shape)
                letter = torch.argmax(output, dim=2)
                sentence_id.append(letter.item())
                #print(letter)
                #print(letter.shape)
                input = self.embed(letter)
                #print(input.shape)
                #input = input.view(1, 1, -1)
                #print(output)
                #print("-" * 50)
                #print(h_n)
                #print(c_n)
                #f
        else:

            for i in range(sentence_length):
                output, (h_n, c_n) = self.rnn(input, (h_n, c_n))
                output = self.output_mapping(output)
                #print(output.shape)

                new_distribution = nn.functional.softmax(output.squeeze() / temperature)

                sample_letter = torch.multinomial(new_distribution, 1)

                sentence_id.append(sample_letter.item())

                #print(letter)
                #print(letter.shape)
                input = self.embed(sample_letter)
                #print(input.shape)
                input = input.view(1, 1, -1)
                #print(input.shape)
                #print(output)
                #print("-" * 50)
                #print(h_n)
                #print(c_n)
                #f

        return sentence_id