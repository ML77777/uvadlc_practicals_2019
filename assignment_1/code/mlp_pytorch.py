"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError

    super(MLP, self).__init__()

    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_classes = n_classes

    self.layers = []

    #Since softmax is in cross_entropy module, only have ReLu activation function which we can do in forward
    #And dont need to keep track of activations functions as backgrad keeps track of gradients
    input_layer = nn.Linear(n_inputs, n_hidden[0], bias=True)
    self.layers.append(input_layer)

    #Hidden layers
    for i,hidden_size in enumerate(n_hidden[:-1]):
      next_hidden_size = n_hidden[i+1]
      hidden_layer = nn.Linear(hidden_size, next_hidden_size,bias = True)
      self.layers.append(hidden_layer)

    #Output layer
    last_layer_size = n_hidden[-1]
    output_layer = nn.Linear(last_layer_size, n_classes,bias=True)
    self.layers.append(output_layer)

    self.torch_layers = nn.ModuleList(self.layers)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError

    for linear_layer in self.layers[:-1]:
      x_tilde = linear_layer(x)
      x = nn.ReLU(x_tilde,inplace=False)
      #x = nn.functional.relu(x_tilde)

    last_layer = self.layers[-1]
    out = last_layer(x)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out
