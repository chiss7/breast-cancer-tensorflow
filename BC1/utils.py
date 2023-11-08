import tensorflow as tf
import numpy as np

class Model:
  def __init__(self, *args):
    self.arch = args[0]
    self.title = f"Model architecture -> input[30], hidden{self.arch}, output[1]"
    # The reason for using the ReLU activation function twice is to introduce non-linearity into the neural network.
    self.activation_functions = [tf.nn.relu, tf.nn.relu, tf.math.sigmoid]
    self.built = False


  """
  Init weights and biases
  args: 
    input_len: number of features of x
  """
  def build(self, input_len):
    # Init w with mean 0, var 1
    w_init = tf.random.normal
    # Init b
    b_init = tf.zeros

    self.weights = []
    self.biases = []

    ## Loop through layers
    for dim in self.arch:
      self.weights.append(tf.Variable(w_init(shape=(dim, input_len))))
      self.biases.append(tf.Variable(b_init(shape=(dim, 1))))
      input_len = dim
    
    ## Final Layer [1 neuron]:
    # Init weights for final layer
    self.weights.append(tf.Variable(w_init(shape=(1, dim))))
    # Init biases for final layer
    self.biases.append(tf.Variable(b_init(shape=(1))))

    self.Variables = self.weights + self.biases


  """
  Feedforward function
  args:
    x: input tensor (30, 426)
  return:
    y: output tensor (1, 426)
  """
  def __call__(self, x):
    # Check if run for first time
    if not self.built:
      self.build(x.shape[0])
      self.built = True

    for w, b, act_f in zip(self.weights, self.biases, self.activation_functions):
      z = tf.matmul(w, x) + b
      a = act_f(z)
      #if i < len(self.arch):
        #a = tf.nn.experimental.stateless_dropout(a, rate=0.5, seed=[1, 0])
      x = a

    return a


  """
  Loss function (Mean Squared Error)
  args:
    y_pred: input tensor 
    y_true: np array
  return:
    loss: output tensor
  """
  def compute_loss(self, y_pred, y_true):
    return tf.reduce_sum((y_pred - y_true) ** 2) / y_true.shape[0]
  

  """
  Acc function
  args:
    y_pred: input tensor 
    y_true: np array
  return:
    acc: output tensor
  """
  def compute_accuracy(self, y_pred, y_true):
    return np.mean(tf.equal(tf.constant(y_true), tf.round(y_pred)))

