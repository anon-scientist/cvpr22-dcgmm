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
from abc                import abstractmethod
from abc                import ABC
import tensorflow       as tf
import numpy            as np
from DCGMM.parsers      import Kwarg_Parser

class Layer(ABC):
  def __init__(self, input, **kwargs):
    self.prev = input

    if input: self.layer_id = self.prev.layer_id + 1
    else    : self.layer_id = 0

    self.next       = None                                    # next layer
    if self.prev: self.prev.next = self                       # concatenate previous with this layer

    self.prefix     = f'L{self.layer_id}_'
    self.parser     = Kwarg_Parser(prefix=self.prefix, **kwargs)
    self.data_type  = self.parser.add_argument('--data_type', type=int  , default=32, choices=[32, 64], help='used data type (float32, int32 or float64, int64) for all calculations and variables (numpy and TensorFlow)')

    if self.data_type == 32: # 32 bit data type
      self.dtype_tf_float, self.dtype_tf_int = tf.float32, tf.int32
      self.dtype_np_float, self.dtype_np_int = np.float32, np.int32
    else:                    # 64 bit data type
      self.dtype_tf_float, self.dtype_tf_int = tf.float64, tf.int64
      self.dtype_np_float, self.dtype_np_int = np.float64, np.int64


  # set external parameter
  def set_parameters(self, **kwargs): pass

  # training and sampling

  # forward propagation
  def forward(self, input_tensor, extra_inputs=None): return input_tensor

  # gives back the per-sample loss!  as subjected to tF.reduce_mean before grad computation!
  def loss(self, input, **kwargs)                   : return -1
  def update_with_grad(self, grads)                 : pass

  # output to next layer.  Is passed the output of forward()
  def get_output(self, fwd_res, **kwargs)           : return fwd_res
  # sampling given a control signbal from higher layer
  def backwards(self, topdown, *args, **kwargs)     : return topdown

  # Testing
  def post_test_step(self, **kwargs)  : pass
  def pre_test_step(self, **kwargs)   : pass
  def evaluate(self, output, **kwargs): return {}

  @abstractmethod
  def get_shape(self)            : pass
  def get_layer_variables(self, **kwargs): return {}
  def is_trainable(self)         : return False
  def reset_layer(self, **kwargs): pass
  def compile(self, **kwargs)    : pass

  # sharpening, need only be re-implemented for folding layers
  def get_target_layer(self): return -1 ;
  def get_reconstruction_weight(self): return -1. ;
  def get_sharpening_iterations(self): return 0 ;
  def get_sharpening_rate(self): return 0.0 ;
  

  # allows to distribute variables and configurations via socket
  def share_variables(self, *args, **kwargs): pass
  def fetch_variables_conf(self)            : pass


  def __str__(self):
    ''' string representation of a layer (print all variables) '''
    max_ = len(max(vars(self).keys(), key=len))
    s = ''
    s += f'Layer: {self.name}\n'
    for k, v in sorted(vars(self).items()):
      if k.startswith('_')       : continue
      s += f' {k:<{max_}}:{v}\n'
    return s


  def set_prev_next(self, prev):
    ''' set previous layer and next layer (of previous layer) by hand '''
    self.prev      = prev
    self.prev.next = self


  def _add_layer_prefix(self, kwargs):
    ''' add the layer prefix if a parameter named "name" exists '''
    if 'name' in kwargs: kwargs['name'] = self.prefix + kwargs['name']


  def _add_dtype(self, kwargs):
    ''' add previously defined dtype_tf_float to all TF variables (if not otherwise stated)'''
    if 'dtype' not in kwargs: kwargs['dtype'] = self.dtype_tf_float


  def constant(self, *args, **kwargs):
    ''' create a layer specific named tensorflow constant '''
    self._add_layer_prefix(kwargs)
    self._add_dtype(kwargs)
    return tf.constant(*args, **kwargs)


  def variable(self, *args, **kwargs):
    ''' create a layer specific named tensorflow variable '''
    self._add_layer_prefix(kwargs)
    self._add_dtype(kwargs)
    return tf.Variable(*args, **kwargs)


  def constant_initializer(self, *args, **kwargs):
    ''' create a layer specific named tensorflow variable '''
    self._add_layer_prefix(kwargs)
    #self._add_dtype(kwargs) # TF2 has no dtype for constant_initializers?
    return tf.constant_initializer(*args, **kwargs)


  def is_layer_type(self, class_name):
    ''' test if the given class_name "is in" the layer class name
    @param class_name: name to test if is in class name string
    '''
    return class_name in self.__class__.__name__


  def get_name(self):
    ''' return the layer name '''
    return self.name

  def pre_train_step(self):
    pass ;

  def post_train_step(self):
    pass ;

  def do_all(self, input_tensor, mask, ys = None):
    self.pre_train_step() 
    self.fwd,self.out,self.raw_loss, self.std_loss = self.do_all_graph(input_tensor, mask, ys=ys)
    self.post_train_step()
    return self.fwd, self.out, self.raw_loss, self.std_loss ;


  @tf.function(autograph=False)
  def do_all_graph(self, input_tensor, mask, ys=None):
      with tf.GradientTape() as g:
        forward_res = self.forward(tf.stop_gradient(input_tensor))
        output_res  = self.get_output(forward_res) 
        raw_loss = self.loss(forward_res, ys=ys) * mask 
        # correct for the mask. If it is a dummy (all ones) then no correction is applied
        # if it has zeroes, then we average only over the samples having mask=1
        loss = tf.reduce_mean(raw_loss)  * self.batch_size / tf.reduce_sum(mask)
      if self.is_trainable():
        vars = self.get_layer_variables() 
        grad = g.gradient(loss, vars) 
        self.update_with_grad(grad) 
      return forward_res, output_res, raw_loss, loss 






