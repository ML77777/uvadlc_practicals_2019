"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
    #raise NotImplementedError

    #Mean 0, std = 0.0001 and W was dl x dl-1 dimension and bias dl dimension
    size_weight = (out_features,in_features)
    size_bias = (out_features,1)
    self.params['weight'] = np.random.normal(loc = 0,scale = 0.0001,size = size_weight)
    self.params['bias'] = np.zeros(size_bias)
    self.grads['weight'] = np.zeros(size_weight)
    self.grads['bias'] = np.zeros(size_bias)

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError

    w = self.params['weight']
    b = self.params['bias']

    out = np.add(np.matmul(w,x.T),b)
    out = out.T
    #out = np.add(np.matmul(x,w.T), b.T)

    self.input = x
    self.out = out

    return out

    ########################
    # END OF YOUR CODE    #
    #######################

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError

    weights = self.params['weight']
    bias = self.params['bias']
    input = self.input

    dx = np.matmul(dout,weights)

    #Update weights
    self.grads["weight"] = np.matmul(dout.T, input)

    #Update bias
    dout_sum = dout.sum(axis=0)
    bias_grads_shape = np.shape(bias)
    self.grads["bias"] = dout_sum.reshape(bias_grads_shape)

    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError

    out = np.maximum(0,x)

    self.input = x
    self.out = out

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError
    # 1 if x > 0 else 0 for the ReLU deriv

    deriv = np.minimum(1,self.out)

    #dx = np.matmul(dout.T,deriv)
    dx = dout * deriv

    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError

    #Log-sum-exp trick preventing numerical problems
    #np.add(np.log(np.sum(np.exp(log_q - a))), a)

    #Prevent overflow with max trick
    x_max = np.amax(x,axis=1,keepdims=True)
    exp_x = np.exp(x - x_max)
    denominator = np.sum(exp_x,axis=1,keepdims=True)
    out = exp_x/denominator

    self.input = x
    self.out = out

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError

    #Softmax has different derivative depending on if indices are the same, which is the case if it is on the diagonal.
    #Thus can use a diagonal matrix to handle both cases
    softmax = self.out
    row,col = np.shape(softmax)
    diagonal_3d = np.zeros((row,col,col))
    diagonal_indices = np.arange(col)
    diagonal_3d[:, diagonal_indices, diagonal_indices] = softmax

    #Piazza
    #np.einsum('i,i->', u, v)  # dot product between two vectors u and v (no index remains)
    #np.einsum('i,j->ij', u, v)  # outer product of two vectors u and v (both indices remain)
    #np.einsum('ij,j->i', W, v)  # matrix vector multiplication
    #np.einsum('i,ij->j', v, W)  # transposed-vector matrix multiplication

    softmax_deriv = diagonal_3d - np.einsum('ij, ik -> ijk', softmax, softmax)
    #softmax_deriv = np.subtract(diagonal_3d - np.dot(softmax, softmax))

    dx = np.einsum('ij, ijk -> ik', dout, softmax_deriv)

    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError
    #Offset to prevent underflow
    self.min_offset = 1e-7

    indv_losses =  np.sum(-1 * y * np.log(x + self.min_offset), axis=1)
    average_losses = np.mean(indv_losses)
    out = average_losses

    self.input = x
    self.out = out

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #raise NotImplementedError

    dx = -1 * np.divide(y, x + self.min_offset) / len(y)

    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
