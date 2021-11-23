''' Delay Layer 
Concatenates current input with input from last two time steps. 
'''


import tensorflow                    as tf
import numpy                         as np
from layer                           import Layer
from utils                           import log
from itertools                       import product
from tensorflow.python.framework.ops import Tensor
import math ;


class Delay_Layer(Layer):

  def __init__(self, input=None, **kwargs):

    Layer.__init__(self, input=input, **kwargs) ;
    parser = self.parser ;

    self.name               = parser.add_argument('--name'      , type=str, default=f'{self.prefix}delay', help='name of the delay layer')
    self.buf_len            = parser.add_argument('--buf_len'   , type=int, default=3,                     help='size of buffer, default=3')
    self.axis               = parser.add_argument('--axis'      , type=int, default=3,                     help='axis at which to concat, default = 3') 
    self.reset              = parser.add_argument('--seq_len'   , type=int, default=10,                    help='reset buf after how many samples?, default = 10')

    log.debug(f'buf_len: {self.buf_len}, axis: {self.axis}, reset: {self.reset}')

    input_shape              = input.get_shape() ;
    self.batch_size          = input_shape[0]
    self.hIn                 = input_shape[1]
    self.wIn                 = input_shape[2]
    self.cIn                 = input_shape[3]

    # compute output size, pad input if required!!
    tmp = [1, 1, 1, 1]
    tmp[self.axis] = tmp[self.axis]*self.buf_len
    self.bOut                = self.batch_size*tmp[0] 
    self.hOut                = self.hIn*tmp[1]
    self.wOut                = self.wIn*tmp[2]
    self.cOut                = self.cIn*tmp[3]

    log.debug(f'{self.name} input shape {input_shape}')
    log.debug(f'{self.name} hOut={self.hOut} wOut={self.wOut} cOut={self.cOut}')

  def compile(self):
    buf = np.zeros((self.bOut, self.hOut, self.wOut, self.cOut), dtype='float32')
    # self.buffer = tf.cast(tf.Variable(initial_value=buf), dtype='float32')
    self.buffer = tf.Variable(initial_value=buf)
    self.counter = tf.Variable(initial_value=0) 
    log.debug(f'dtype buffer: {self.buffer.dtype}')
    log.debug(f'buffer shape: {self.buffer.shape} -> ({self.bOut} x {self.hOut} x {self.wOut} x {self.cOut})')

  @tf.function(autograph=False)
  def forward(self, input_tensor, extra_inputs = None):
    tf.cond(tf.math.equal(self.counter, 0), lambda: tf.compat.v1.assign(self.buffer, tf.math.multiply(self.buffer, 0)), lambda: tf.compat.v1.assign(self.buffer, tf.math.multiply(self.buffer, 1)))

    ''' remove oldest sample from buffer and add the current one to the end of it '''
    newbuf = tf.gather(self.buffer, np.arange(int(self.buffer.shape[self.axis]/self.buf_len), self.buffer.shape[self.axis]), axis=self.axis)
    tf.compat.v1.assign(self.buffer, tf.concat([newbuf, input_tensor], axis=self.axis))
    log.debug(f'buffer shape: {self.buffer.shape}, dtype: {self.buffer.dtype}')
    
    tf.compat.v1.assign(self.counter, (self.counter + 1) % self.reset)
    
    return self.buffer
  
  # TODO FIX this! For the moment only a simple placeholder doing nothing and sampling back the correct shape
  # later, the model needs to query this intellgiently to obtain the true sampled sequence
  def backwards(self, topdown, *args, **kwargs):
    return tf.zeros([self.batch_size, self.hIn, self.wIn, self.cIn],dtype=self.dtype_tf_float) ;
  


  def get_shape(self): return [self.bOut, self.hOut, self.wOut, self.cOut]

