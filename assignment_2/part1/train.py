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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM
#from dataset import PalindromeDataset
#from vanilla_rnn import VanillaRNN
#from lstm import LSTM
import matplotlib.pyplot as plt

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == "RNN":
        model = VanillaRNN(config.input_length, config.input_dim,  config.num_hidden, config.num_classes, config.batch_size, config.device)
    else:
        model = LSTM(config.input_length, config.input_dim,  config.num_hidden, config.num_classes, config.batch_size, config.device)

    device = model.device
    model.to(device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr = config.learning_rate)

    list_train_acc = []
    list_train_loss = []
    acc_average = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        #model.to(device)

        output = model(batch_inputs)
        loss = criterion(output,batch_targets)
        model.zero_grad()
        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        # it clips gradient norm of an iterable of parameters, so the gradients are normalized w.r.t. to the max_norm
        # Thus is will limit the size and get reasonably gradients as opposed to very large gradients.
        # This handles the case of exploding gradients as with each layer the gradient can get amplified.
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        # Add more code here ...
        optimizer.step()

        #Loss is computed above
        #loss = np.inf   # fixme
        #accuracy = 0.0  # fixme

        number_predictions = torch.argmax(output, dim=1)
        result = number_predictions == batch_targets
        accuracy = result.sum().item() / len(batch_targets)


        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

            #Add to list
            list_train_acc.append(accuracy)
            list_train_loss.append(loss)

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

    if not config.overview_length:
        eval_steps = list(range(0,config.train_steps+10,10))
        plt.plot(eval_steps, list_train_acc, label="Train accuracy")
        plt.xlabel("Evaluation step")
        plt.ylabel("Accuracy")
        plt.title("Training accuracies", fontsize=18, fontweight="bold")
        plt.legend()
        # plt.savefig('accuracies.png', bbox_inches='tight')
        plt.show()

        plt.plot(eval_steps, list_train_loss, label="Train loss")
        plt.xlabel("Evaluation step")
        plt.ylabel("Loss")
        plt.title("Train loss", fontsize=18, fontweight="bold")
        #plt.legend()
        # plt.savefig('loss.png', bbox_inches='tight')
        plt.show()

    return (list_train_acc,list_train_loss)

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
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--overview_length', type=int, default="0", help="Plot the accuracy curves of different Palindromes lengths")

    config = parser.parse_args()

    config.model_type = "LSTM"
    config.input_length = 20
    config.learning_rate = 0.01

    #Seeds for for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if 'cuda' in config.device.lower() and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config.overview_length:
        overview = []

        eval_steps = list(range(0, config.train_steps + 10, 10))

        for i in range(4,50,5):
            config.input_length = i
            list_train_acc,list_train_loss = train(config)
            #info = (i+1,train_acc,train_loss)
            #overview.append(info)

            plt.plot(eval_steps, list_train_acc, label=str(i+1))

        plt.xlabel("Evaluation step")
        plt.ylabel("Accuracy")
        plt.title("Training accuracies of different Palindromes lengths", fontsize=18, fontweight="bold")
        plt.legend(title="Lengths")
        # plt.savefig('accuracies.png', bbox_inches='tight')
        plt.show()

        #plt.plot(eval_steps, list_train_loss, label="Train loss")
        #plt.xlabel("Evaluation step")
        #plt.ylabel("Loss")
        #plt.title("Train loss", fontsize=18, fontweight="bold")
        #plt.legend()
        # plt.savefig('loss.png', bbox_inches='tight')
        #plt.show()
    else:
        # Train the model (once)
        train(config)

