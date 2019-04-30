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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
from part2.dataset import TextDataset
from part2.model import TextGenerationModel

################################################################################

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file,config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    #print(dataset._char_to_ix) vocabulary changes, but batches are the same sentence examples.

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,config.lstm_num_hidden, config.lstm_num_layers, config.device)  # fixme

    device = model.device
    model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    #criterion2 = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr = config.learning_rate)
    print("Len dataset:", len(dataset))
    print("Amount of steps for dataset:",len(dataset)/config.batch_size)

    #amount_epochs =  round(config.train_steps / round(len(dataset)/config.batch_size))
    current_step = -1
    not_max = True

    while not_max:

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            current_step += 1
            #######################################################
            # Add more code here ...

            #List of indices from word to ID, that is in dataset for embedding
            #indices = ..
            #model.forward(indices)
            #print(batch_inputs[0])
            #print(len(batch_inputs))  30 Seq_length, so many tensors
            #print(len(batch_inputs[0]))  64 batch_size, length of a tensor
            #print(batch_targets)
            #print(len(batch_targets))
            embed = model.embed #nn.Embedding(dataset.vocab_size, config.lstm_num_hidden)
            #print(embed(batch_inputs[0]))

            #print(batch_inputs)
            #print(torch.stack(batch_inputs).shape)
            all_embed = []
            sentence = []
            for batch_letter in batch_inputs:
                embedding = embed(batch_letter)
                all_embed.append(embedding)
                sentence.append(batch_letter[0].item())
            all_embed = torch.stack(all_embed)
            #print(all_embed.shape)
            #print(len(batch_targets))
            #print(len(batch_targets[0]))

            #print(dataset.convert_to_string(sentence))

            sentence = []
            for batch_letter in batch_targets:
                sentence.append(batch_letter[0].item())
            #print(dataset.convert_to_string(sentence))

            outputs = model(all_embed) #[30,64,87] dimension for now



            #######################################################

            #loss = np.inf   # fixme
            accuracy = 0.0  # fixme


            #Method 1, turn 3d into 2d tensor
            outputs_2 = outputs.view(-1, dataset.vocab_size)
            #print(outputs_2.shape)
            #print(batch_targets)
            batch_targets = torch.stack(batch_targets)
            batch_targets_2 = batch_targets.view(-1)
            batch_targets_2 = batch_targets_2.to(device)
            #print(batch_targets_2.shape)

            loss = criterion(outputs_2, batch_targets_2)

            #Method 2, ensuring that the prediction dim are batchsize x vocab_size x sequence length and targets: batchsize x sequence length
            #Same loss value, but slower
            #reformed_outputs = []
            #for second_dim in range(outputs.shape[1]):
            #    a_sequence = []
            #    for first_dim in range(outputs.shape[0]):
            #        a_sequence.append(outputs[first_dim,second_dim,:])
            #    reformed_outputs.append(torch.t(torch.stack(a_sequence)))
            #reformed_outputs = torch.stack(reformed_outputs)
            #print(reformed_outputs.shape)

            #loss2 = criterion2(reformed_outputs,torch.t(batch_targets))
            #loss = criterion(reformed_outputs, torch.t(batch_targets))
            #for o in outputs:

            #print(loss)

            #break

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            sentence_0 = outputs[:,0,:].argmax(dim=1)
            #print(outputs[:,0,:])
            #print(outputs[:,0,:].argmax(dim=1))
            #print(batch_targets.shape)
            #print(batch_targets)

            #Predicted characters of example 1
            #print(sentence_0)
            #print(dataset.convert_to_string(sentence_0.tolist()))

            number_predictions = torch.argmax(outputs, dim=2)
            result = number_predictions == batch_targets
            accuracy = result.sum().item() / (batch_targets.shape[0] * batch_targets.shape[1])


            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), current_step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if step == config.sample_every:
                # Generate some sentences by sampling from the model
                pass

            if current_step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                not_max = False
                break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    #parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    ############################################################################################################
    #Change back later
    #parser.add_argument('--txt_file', type=str, default="./assets/book_NL_darwin_reis_om_de_wereld.txt", help="Path to a .txt file to train on")
    parser.add_argument('--txt_file', type=str, default="./assets/book_EN_grimms_fairy_tails.txt", help="Path to a .txt file to train on")
    #parser.add_argument('--txt_file', type=str, default="./assets/book_EN_democracy_in_the_US.txt", help="Path to a .txt file to train on")

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

    parser.add_argument('--train_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    config.train_steps = 9000
    #Same sentence examples.
    torch.manual_seed(42)
    np.random.seed(42)
    if 'cuda' in config.device.lower() and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Train the model
    train(config)
