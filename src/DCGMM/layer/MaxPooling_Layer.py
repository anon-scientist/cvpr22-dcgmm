# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
''' Max Pooling Layer
Standard CNN max pooling, but together with backwards sampling.
If kernel size is not a divisor of input tensor h/w, input is zero-padded.
This is not implemented physically but by manipulating the lookup structures
'''


import tensorflow                    as tf
import numpy                         as np
from .                               import Layer
from DCGMM.utils                     import log
from itertools                       import product
from tensorflow.python.framework.ops import Tensor
import math


class MaxPooling_Layer(Layer):

  def __init__(self, input=None, **kwargs):

    Layer.__init__(self, input=input, **kwargs)
    parser = self.parser

    self.name               = parser.add_argument('--name'       , type=str, default=f'{self.prefix}maxPooling', help='name of the maxPooling layer')

    input_shape              = input.get_shape()
    self.batch_size          = input_shape[0]
    self.hIn                 = input_shape[1]
    self.wIn                 = input_shape[2]
    self.cIn                 = input_shape[3]

    self.sharpening_rate       = self.parser.add_argument('--sharpening_rate'      , type=float, default=.1                     , help='if sampling is active, use sharpening rate to improve samples with gradient')
    self.sharpening_iterations = self.parser.add_argument('--sharpening_iterations', type=int  , default=100                    , help='number of sharpening iterations')
    self.target_layer          = self.parser.add_argument('--target_layer',          type=int  , default=-1                    , help='target GMM layer index for sharpening')
    self.kernel_size_y         = parser.add_argument('--kernel_size_y', type=int, default=self.hIn                  , help='kernel width y')
    self.kernel_size_x         = parser.add_argument('--kernel_size_y', type=int, default=self.wIn                  , help='kernel height x')
    self.kernel_size_t         = parser.add_argument('--kernel_size_t', type=int, default=1                  , help='kernel size in temporal dimension, assuming that temporal dim is folde"d into the channel dimension')
    self.kernel_size_x             = self.kernel_size_x * self.kernel_size_t 

    self.wIn                 = self.wIn * self.kernel_size_t
    self.cIn                 = self.cIn // self.kernel_size_t

    self.stride_y             = parser.add_argument('--stride_y'   , type=int, default=1                         , help='stride y')
    self.stride_x             = parser.add_argument('--stride_x'   , type=int, default=1                         , help='stride x')
    self.stride_x             = self.stride_x * self.kernel_size_t 

    self.sampling_mode        = parser.add_argument('--sampling_mode'   , type=str, default="dense"                        , help='dense or sparse')

    # comopute output size, pad input if required!!
    self.hOut                = 1 + math.ceil((self.hIn - self.kernel_size_y) / self.stride_y)
    self.wOut                = 1 + math.ceil((self.wIn - self.kernel_size_x) / self.stride_x)
    self.cOut                = self.cIn

    log.debug(f'{self.name} input shape {input_shape}')
    log.debug(f'{self.name} kernelSizeY={self.kernel_size_y} kernelSizeX={self.kernel_size_x} strideY={self.stride_y} strideX={self.stride_x}')
    log.debug(f'{self.name} hOut={self.hOut} wOut={self.wOut} cOut={self.cOut}')

  def get_target_layer(self): return self.target_layer 
  def get_sharpening_iterations(self): return self.sharpening_iterations 
  def get_sharpening_rate(self): return self.sharpening_rate



  def compile(self):
    ''' build tf variables, mainly lookups '''
    # pre-compute constants for pooling
    # -- for collecting input tensor values via tf.gather such that max can be taken
    lookupShape = [self.hOut, self.wOut, self.cOut, self.kernel_size_y, self.kernel_size_x]
    self.np_lookupArray = np.zeros(lookupShape, dtype=np.int64)
    # -- remember indices of values to null before max taking since they come from outside the input and contain corrupted values. But should contain zeroes
    self.np_zeroMask =  np.ones(lookupShape, dtype=self.dtype_np_float)

    # -- for (up-)sampling
    self.np_invArray    = np.zeros([self.hIn, self.wIn, self.cIn])


    # construct constants
    # -- forward lookup
    # ---- loop over grid positions of filter windows in input
    for h, w, c in product(range(self.hOut), range(self.wOut), range(self.cOut)):
      # ---- loop over input pixels IN filter windows at a certain position
      for inPatchY, inPatchX in product(range(self.kernel_size_y), range(self.kernel_size_x)):
        inPatchStartY                                 = h * self.stride_y
        inPatchStartX                                 = w * self.stride_x
        inC                                           = c % self.cIn
        inY                                           = inPatchStartY + inPatchY
        inX                                           = inPatchStartX + inPatchX

        if inY >= self.hIn or inX >= self.wIn:
          self.np_lookupArray[h, w, c, inPatchY, inPatchX] = 0
          self.np_zeroMask[h, w, c, inPatchY, inPatchX] = 0
        else:
          self.np_lookupArray[h, w, c, inPatchY, inPatchX] = self.wIn * self.cIn * inY + self.cIn * inX + inC
    self.lookupArray = tf.constant(self.np_lookupArray.reshape((self.hOut * self.wOut * self.cOut * self.kernel_size_y * self.kernel_size_x)))
    self.zeroMaskArray = tf.constant(self.np_zeroMask.reshape(1, -1))

    # -- sampling lookup
    self.np_inv_arr            = np.zeros([self.hIn * self.wIn * self.cIn], dtype=np.int64)

    for inIndex, (h, w, c) in enumerate(product(range(self.hIn), range(self.wIn), range(self.cIn))):
      outY                     = h // self.kernel_size_y
      outX                     = w // self.kernel_size_x
      outC                     = c
      outIndex                 = outY * self.wOut * self.cOut + outX * self.cOut + outC
      self.np_inv_arr[inIndex] = outIndex

    self.invArr             = tf.constant(self.np_inv_arr)

    shufflingMask = np.ones([self.hOut*self.wOut, self.kernel_size_x*self.kernel_size_y])*-1. ;
    patchSize = self.kernel_size_x * self.kernel_size_y ;
    
    # todo correct for fact that border patches may have their ones outside the input tensor

    # two ways of sampling throught maxcpooling; a) repeat topdown value to all pixels in patch  b) repat topdown value to single, random position in patch, otherwise 0.0
    # create a tensor that serves as mask for the generated input in case of a)
    for c in range(0,self.hOut*self.wOut):
      offset = c % patchSize ;
      shufflingMask [c,offset] = 1.0 ;

    #print ("SH=",shufflingMask)
    self.shuffling_mask = tf.constant(shufflingMask,dtype=self.dtype_tf_float) 
    



  @tf.function(autograph=False)
  def forward(self, input_tensor, extra_inputs = None):
    ''' try a simple approach analogous to folding layer: use gather to copy all 2x2 patches to a continuous channel dimension '''

    foldedTensor     = tf.reshape(input_tensor, (self.batch_size, -1))
    foldedTensor     = tf.gather(foldedTensor, self.lookupArray, axis=1)
    #print (f"folded Tensor = {foldedTensor.shape}")
    foldedTensorMasked = foldedTensor * self.zeroMaskArray
    foldedTensor     = tf.reshape(foldedTensor, (-1, self.hOut, self.wOut, self.cOut, self.kernel_size_y * self.kernel_size_x))
    maxOp            = tf.reduce_max(foldedTensor, axis=4)
    #log.debug(f'MaxP Put: {maxOp}')
    return maxOp


  def backwards(self, topdown, *args, **kwargs):

    log.debug(f"hin, win, cin={self.hIn}, {self.wIn}, {self.cIn}")
    log.debug(f"topdown shape={topdown.shape}")
    tmp =  tf.reshape(tf.gather(tf.reshape(topdown, (-1, self.hOut * self.wOut * self.cOut)), self.invArr, axis=1), (-1, self.hIn, self.wIn // self.kernel_size_t, self.cIn * self.kernel_size_t))
    log.debug(f"to lower shape = {tmp.shape}")

    # TODO does this work with sp.temporal pooling (kernel_size_t > 1) ?
    if self.sampling_mode == "sparse":
      mask1 = tf.random.shuffle(self.shuffling_mask) ; # hOut*wOut,ksX*ksY
      mask2 = tf.reshape(mask1,[1,self.hOut,self.wOut,self.kernel_size_y,self.kernel_size_x])  # --> 1,hOut,wOut, ksY,ksX
      mask3 = tf.transpose(mask2,[0,1,3,2,4]) ;  # --> 1, hOut, ksY, wOut, ksX
      mask4 = tf.reshape(mask3,[1,self.hOut*self.kernel_size_y, self.wOut*self.kernel_size_x, 1]) # --> 1, hOut*ksY, wOut*ksX

      return  tmp*mask4[:,0:self.hIn, 0:self.wIn,:] ;
    else:
      return  tmp





  def get_shape(self): return [self.batch_size, self.hOut, self.wOut, self.cOut]


  def is_trainable(self):
    return False 

