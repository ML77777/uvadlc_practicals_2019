"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
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

    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.amount_hidden = len(n_hidden)
    self.n_classes = n_classes
    self.learning_rate = 0.05
    #self._batch_size =

    #Store layers here in the form of tuples, feed forward and activation
    self.layers = []

    #Input layer
    input_layer = LinearModule(n_inputs, n_hidden[0])
    input_layer_a = ReLUModule()
    tuple = (input_layer,input_layer_a)
    self.layers.append(tuple)

    #Hidden layers
    for i,hidden_size in enumerate(n_hidden[:-1]):
      next_hidden_size = n_hidden[i+1]
      hidden_layer = LinearModule(hidden_size, next_hidden_size)
      hidden_layer_a = ReLUModule()
      tuple = (hidden_layer, hidden_layer_a)
      self.layers.append(tuple)

    #Output layer
    last_layer_size = n_hidden[-1]
    output_layer = LinearModule(last_layer_size, n_classes)
    output_layer_a = SoftMaxModule()
    tuple = (output_layer,output_layer_a)
    self.layers.append(tuple)

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

    for pre_layer,activation in self.layers:
      x_tilde = pre_layer.forward(x)
      x = activation.forward(x_tilde)

    out = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError

    for pre_layer,activation in reversed(self.layers):
      dout = activation.backward(dout)
      dout = pre_layer.backward(dout)

    ########################
    # END OF YOUR CODE    #
    #######################

    return
