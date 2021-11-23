import tensorflow as tf
import numpy      as np
from .            import Layer
from DCGMM.utils  import log
from itertools    import product

class Folding_Layer(Layer):
  ''' a (un)folding (convolution) layer '''

  def __init__(self, input=None, **kwargs):
    Layer.__init__(self, input=input, **kwargs)

    self.name                   = self.parser.add_argument('--name'                 , type=str  , default=f'{self.prefix}folding', help='name of the folding layer')
    self.batch_size            = self.parser.add_argument('--batch_size'         , type=int  , default=100                                  , help='used batch size', prefix=False)

    self.sampling_batch_size    = self.parser.add_argument('--sampling_batch_size'  , type=int  , default=100                    , help='sampling batch size')
    self.patch_height           = self.parser.add_argument('--patch_height'         , type=int  , default=-1                     , help='patch height')
    self.patch_width            = self.parser.add_argument('--patch_width'          , type=int  , default=-1                     , help='patch width')
    self.stride_y               = self.parser.add_argument('--stride_y'             , type=int  , default=-1                     , help='stride y')
    self.stride_x               = self.parser.add_argument('--stride_x'             , type=int  , default=-1                     , help='stride x')
    self.reconstruction_weight        = self.parser.add_argument('--reconstruction_weight', type=float, default=.1                     , help='if sampling is active, use sharpening rate to improve samples with gradient')
    self.sharpening_rate        = self.parser.add_argument('--sharpening_rate'      , type=float, default=.1                     , help='if sampling is active, use sharpening rate to improve samples with gradient')
    self.sharpening_iterations  = self.parser.add_argument('--sharpening_iterations', type=int  , default=100                    , help='number of sharpening iterations')
    self.target_layer           = self.parser.add_argument('--target_layer',          type=int  , default=-1                    , help='target GMM layer index for sharpening')

    input_shape                 = self.prev.get_shape()
    log.debug(f'input shape{input_shape}')
    self.batch_size             = input_shape[0]
    self.w_in                   = input_shape[1]
    self.h_in                   = input_shape[2]
    self.c_in                   = input_shape[3]

    if self.patch_height == -1: self.patch_height = self.h_in
    if self.patch_width  == -1: self.patch_width  = self.w_in
    if self.stride_y     == -1: self.stride_y     = self.h_in
    if self.stride_x     == -1: self.stride_x     = self.w_in

    log.debug(f'path: {self.patch_height}x{self.patch_width}, stride: {self.stride_y}x{self.stride_x}')

    self.h_out                  = int((self.h_in - self.patch_height) / (self.stride_y) + 1)
    self.w_out                  = int((self.w_in - self.patch_width)  / (self.stride_x) + 1)
    self.c_out                  = self.patch_height * self.patch_width * self.c_in
    self.output_size            = self.h_out * self.w_out * self.c_out




  def get_shape(self):
    return [self.batch_size, self.h_out, self.w_out, self.c_out]

  def get_target_layer(self): return self.target_layer 
  def get_reconstruction_weight(self): return self.reconstruction_weight 
  def get_sharpening_iterations(self): return self.sharpening_iterations 
  def get_sharpening_rate(self): return self.sharpening_rate


  def compile(self, **kwargs):

    self.indicesOneSample  = np.zeros([int(self.h_out * self.w_out * self.c_out)], dtype=self.dtype_np_int)  # for forward transmission
    # for backward transmission (sampling)
    mapCorr        = np.zeros([1, self.h_in, self.w_in, 1])
    indexArr       = np.zeros([self.sampling_batch_size, self.h_out * self.w_out * self.c_out],dtype=np.int32)
    # create forward and backward indices arrays for use with gather (forward) and scatter (backward) loop over all filter positions in output layer. compute unique index of corresponding pixel in input tensor, and fill arrays
    for outIndex, (outY, outX, outC) in enumerate(product(range(self.h_out), range(self.w_out), range(self.c_out))):
      inFilterY                       = outY * self.stride_y
      inFilterX                       = outX * self.stride_x
      inC                             = outC % self.c_in
      inCFlatIndex                    = outC // self.c_in
      inY                             = inFilterY + inCFlatIndex // self.patch_width
      inX                             = inFilterX + inCFlatIndex % self.patch_width
      inIndex                         = inY * self.w_in * self.c_in + inX * self.c_in + inC
      self.indicesOneSample[outIndex] = inIndex
      indexArr[:, outIndex]           = inIndex
      if inC == 0: mapCorr[0, inY, inX, 0] += 1
    indexArr     += np.array([ [i * self.h_in * self.w_in * self.c_in] for i in range(self.sampling_batch_size) ])
    self.indexArr = tf.constant(indexArr, dtype=self.dtype_tf_int)
    self.mapCorr  = tf.constant(mapCorr, dtype= self.dtype_tf_float)
    acc_shape     = (self.sampling_batch_size * self.h_in * self.w_in * self.c_in)
    self.acc      = self.variable(initial_value=np.zeros(acc_shape), dtype=self.dtype_tf_float, name='acc')


  @tf.function(autograph=False)
  def forward(self, input_tensor):
    ''' transform all samples at the same time by axis=1 arg to gather '''
    log.debug(f'input_shape Folding: {input_tensor.shape}')
    gatherRes  = tf.gather(tf.reshape(input_tensor, (-1, self.h_in * self.w_in * self.c_in)), self.indicesOneSample, axis=1)
    convert_op = tf.reshape(gatherRes, (-1, self.w_out, self.h_out, self.c_out))
    return convert_op


  def backwards(self, topdown, *args, **kwargs):
    self.acc.assign(self.acc * 0.0)
    self.acc.scatter_add(
      tf.IndexedSlices(tf.reshape(topdown, (-1,)),
      tf.cast(tf.reshape(self.indexArr, -1), self.dtype_tf_int))
    )
    backProj    = tf.reshape(self.acc, (self.sampling_batch_size, self.h_in, self.w_in, self.c_in))
    sampling_op = backProj / self.mapCorr
    return sampling_op



