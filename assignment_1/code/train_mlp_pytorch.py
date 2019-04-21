"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch.nn as nn
from torch import optim

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  raise NotImplementedError

  final_predictions_indices = predictions.argmax(axis=1)
  target_indices = targets.argmax(axis=1)
  bool_matrix = final_predictions_indices == target_indices
  tp_tn = bool_matrix.sum()
  accuracy = tp_tn / np.shape(targets)[0]

  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  #raise NotImplementedError

  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
  #x_train_batch, y_train_batch = cifar10['train'].next_batch(5)
  train_data = cifar10['train']
  x_test, y_test = cifar10['test'].images, cifar10['test'].labels

  nsamples_x,channels,y_size,x_size = np.shape(x_test)
  nsamples_y,n_classes = np.shape(y_test)
  input_per_image = y_size * x_size * channels

  MLP_classifier = MLP(input_per_image, dnn_hidden_units, n_classes)

  cross_entropy_loss = nn.CrossEntropyLoss()

  #Evaluation
  list_train_acc = []
  list_test_acc = []
  list_train_loss = []
  list_test_loss = []

  #Reshape here as we do multiple test while training
  x_test = x_test.reshape((nsamples_x, input_per_image))
  x_test = torch.Tensor(x_test)

  optimizer = optim.SGD(MLP_classifier.parameters(),FLAGS.learning_rate)

  for step in range(FLAGS.max_steps):

    #Get batch and reshape for input
    x_train_batch, y_train_batch = train_data.next_batch(FLAGS.batch_size)
    x_train_batch = x_train_batch.reshape((FLAGS.batch_size, input_per_image))
    x_train_batch = torch.Tensor(x_train_batch,requires_grad = True)
    y_train_batch = torch.Tensor(y_train_batch)

    #Feed forward, get loss and gradient of the loss function and backpropagate.
    output = MLP_classifier.forward(x_train_batch)
    break

    train_loss = cross_entropy_loss(output,y_train_batch)
    MLP_classifier.zero_grad()
    train_loss.backward()
    optimizer.step()
    '''
    train_loss = cross_entropy_loss.forward(output, y_train_batch)
    loss_gradient = cross_entropy_loss.backward(output, y_train_batch)
    MLP_classifier.backward(loss_gradient)

    #Gradients are defined in each layer now, update the weights with it
    for pre_layer,activation in MLP_classifier.layers:
      gradient_w = pre_layer.grads['weight']
      gradient_b = pre_layer.grads['bias']
      pre_layer.params['weight'] = pre_layer.params['weight'] - (FLAGS.learning_rate * gradient_w)
      pre_layer.params['bias'] = pre_layer.params['bias'] - (FLAGS.learning_rate * gradient_b)


    if (step % FLAGS.eval_freq) == 0 or (step == FLAGS.max_steps -1):
      output_test = MLP_classifier.forward(x_test)
      test_acc = accuracy(output_test,y_test)
      test_loss = cross_entropy_loss.forward(output_test,y_test)
      list_test_acc.append(test_acc)
      list_test_loss.append(test_loss)

      train_acc = accuracy(output,y_train_batch)
      list_train_loss.append(train_loss)
      list_train_acc.append(train_acc)

  #Print and plot results
  print(list_test_acc)
  print(list_test_loss)
  steps_x = range(len(list_test_acc))
  plt.plot(steps_x,list_test_acc,label="Test accuracy")
  plt.plot(steps_x, list_train_acc, label="Train accuracy")
  plt.xlabel("Step")
  plt.ylabel("Accuracy")
  plt.title("Train and test accuracies", fontsize=18, fontweight="bold")
  plt.legend()
  #plt.savefig('accuracies.png', bbox_inches='tight')
  plt.show()

  plt.plot(steps_x, list_test_loss, label="Test loss")
  plt.plot(steps_x, list_train_loss, label="Train loss")
  plt.xlabel("Step")
  plt.ylabel("Loss")
  plt.title("Train and test loss", fontsize=18, fontweight="bold")
  plt.legend()
  #plt.savefig('loss.png', bbox_inches='tight')
  plt.show()
  '''
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()