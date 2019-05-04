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
import matplotlib.pyplot as plt
import pickle

################################################################################

def train(config):

    # Initialize the device which to run the model on
    #device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file,config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    #print(dataset._char_to_ix) vocabulary order changes, but batches are same sentence examples with the seeds earlier.

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,config.lstm_num_hidden, config.lstm_num_layers, config.device)  # fixme

    device = model.device
    model = model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr = config.learning_rate)
    print("Len dataset:", len(dataset))
    print("Amount of steps for dataset:",len(dataset)/config.batch_size)

    current_step = 0
    not_max = True

    list_train_acc = []
    list_train_loss = []
    acc_average = []
    loss_average = []

    file = open("sentences.txt", 'w', encoding='utf-8')

    '''
    file_greedy = open("sentences_greedy.txt",'w',encoding='utf-8')
    file_tmp_05 = open("sentences_tmp_05.txt", 'w', encoding='utf-8')
    file_tmp_1 = open("sentences_tmp_1.txt", 'w', encoding='utf-8')
    file_tmp_2 = open("sentences_tmp_2.txt", 'w', encoding='utf-8')
    '''

    while not_max:

        for (batch_inputs, batch_targets) in data_loader:

            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...

            #List of indices from word to ID, that is in dataset for embedding
            #Embedding lookup
            embed = model.embed #Embeding shape(dataset.vocab_size, config.lstm_num_hidden)

            #Preprocess input to embeddings to give to LSTM all at once
            all_embed = []
            #sentence = []
            for batch_letter in batch_inputs:
                batch_letter_to = batch_letter.to(device) #torch.tensor(batch_letter,device = device)
                embedding = embed(batch_letter_to)
                all_embed.append(embedding)

                #sentence.append(batch_letter_to[0].item())
            all_embed = torch.stack(all_embed)

            #Print first example sentence of batch along with target
            #print(dataset.convert_to_string(sentence))
            #sentence = []
            #for batch_letter in batch_targets:
            #    sentence.append(batch_letter[0].item())
            #print(dataset.convert_to_string(sentence))

            all_embed  = all_embed.to(device)
            outputs = model(all_embed) #[30,64,vocab_size] 87 last dimension for fairy tails

            #######################################################

            #loss = np.inf   # fixme
            #accuracy = 0.0  # fixme

            #For loss: ensuring that the prediction dim are batchsize x vocab_size x sequence length and targets: batchsize x sequence length
            batch_first_output = outputs.transpose(0, 1).transpose(1, 2)
            batch_targets = torch.stack(batch_targets).to(device)
            loss = criterion(batch_first_output, torch.t(batch_targets))

            #Backpropagate
            model.zero_grad()
            loss.backward()
            loss = loss.item()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            #Accuracy
            number_predictions = torch.argmax(outputs, dim=2)
            result = number_predictions == batch_targets
            accuracy = result.sum().item() / (batch_targets.shape[0] * batch_targets.shape[1])

            ''''
            #Generate sentences for all settings on every step
            sentence_id = model.generate_sentence(config.gsen_length, -1)
            sentence = dataset.convert_to_string(sentence_id)
            #print(sentence)
            file_greedy.write( (str(current_step) + ": " + sentence + "\n"))

            sentence_id = model.generate_sentence(config.gsen_length, 0.5)
            sentence = dataset.convert_to_string(sentence_id)
            #print(sentence)
            file_tmp_05.write( (str(current_step) + ": " + sentence + "\n"))

            sentence_id = model.generate_sentence(config.gsen_length, 1)
            sentence = dataset.convert_to_string(sentence_id)
            #print(sentence)
            file_tmp_1.write( (str(current_step) + ": " + sentence + "\n"))

            sentence_id = model.generate_sentence(config.gsen_length, 2)
            sentence = dataset.convert_to_string(sentence_id)
            #print(sentence)
            file_tmp_2.write( (str(current_step) + ": " + sentence + "\n"))
            '''

            if config.measure_type == 2:
                acc_average.append(accuracy)
                loss_average.append(loss)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if current_step % config.print_every == 0:

                # Average accuracy and loss over the last print every step (5 by default)
                if config.measure_type == 2:
                    accuracy = sum(acc_average) / config.print_every
                    loss = sum(loss_average) / config.print_every
                    acc_average = []
                    loss_average = []

                # Either accuracy and loss on the print every interval or the average of that interval as stated above
                list_train_acc.append(accuracy)
                list_train_loss.append(loss)

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), current_step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))
            elif config.measure_type == 0:
                # Track accuracy and loss for every step
                list_train_acc.append(accuracy)
                list_train_loss.append(loss)

            if current_step % config.sample_every == 0:
                # Generate sentence
                sentence_id = model.generate_sentence(config.gsen_length, config.temperature)
                sentence = dataset.convert_to_string(sentence_id)
                print(sentence)
                file.write((str(current_step) + ": " + sentence + "\n"))

            if current_step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                not_max = False
                break

            current_step += 1

    # Close the file and make sure sentences en measures are saved
    file.close()
    pickle.dump((list_train_acc, list_train_loss), open("loss_and_train.p", "wb"))

    #Plot
    print(len(list_train_acc))

    if config.measure_type == 0:
        eval_steps = list(range(config.train_steps + 1))  # Every step Acc
    else: #
        eval_steps = list(range(0, config.train_steps + config.print_every, config.print_every))

    if config.measure_type == 2:
        plt.plot(eval_steps[:-1], list_train_acc[1:], label="Train accuracy")
    else:
        plt.plot(eval_steps, list_train_acc, label="Train accuracy")

    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy LSTM", fontsize=18, fontweight="bold")
    plt.legend()
    # plt.savefig('accuracies.png', bbox_inches='tight')
    plt.show()

    if config.measure_type == 2:
        plt.plot(eval_steps[:-1], list_train_loss[1:], label="Train loss")
    else:
        plt.plot(eval_steps, list_train_loss, label="Train loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training loss LSTM", fontsize=18, fontweight="bold")
    plt.legend()
    # plt.savefig('loss.png', bbox_inches='tight')
    plt.show()
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

    parser.add_argument('--measure_type', type=int, default="2", help="Track accuracy and loss on every step (0), every print step (1) or every print step take avrerage over those intervals (2)")
    parser.add_argument('--temperature', type=float, default="-1",help="Temperature parameter value, if smaller or equal to zero, then greedy sampling, else random sampling with this value")
    parser.add_argument('--gsen_length', type=int, default="30",help="Length of the sentence to generate")

    config = parser.parse_args()

    #config.device = 'cpu'
    config.train_steps = 50000
    #config.sample_every = 10000
    #config.print_every = 10

    #Same sentence examples.
    torch.manual_seed(42)
    np.random.seed(42)
    if 'cuda' in config.device.lower() and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #favorite_color = pickle.load(open("loss_and_train.p", "rb"))

    # Train the model
    train(config)
