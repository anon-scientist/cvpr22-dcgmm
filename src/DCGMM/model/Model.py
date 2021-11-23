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
import tensorflow             as tf
import numpy                  as np
from abc                      import abstractmethod
from abc                      import ABC
from DCGMM.utils              import log
from DCGMM.checkpoint_manager import Checkpoint_Manager
from DCGMM.parsers            import Kwarg_Parser

class Model(ABC):
  
  def __init__(self, **kwargs):
    ''' primarily reads arguments from kwargs using the kwargsParser class. Semantics are similar to argparse. '''
   
    self.parser             = Kwarg_Parser(**kwargs)

    self.data_type          = self.parser.add_argument('--data_type' , type=int  , default=32 , choices=[32, 64], help='used data type (float32, int32 or float64, int64) for all calculations and variables (numpy and TensorFlow)')
    self.w                  = self.parser.add_argument('--w'         , type=int  , default=-1                   , help='NHWC parameter, supplied by Experiment')
    self.h                  = self.parser.add_argument('--h'         , type=int  , default=-1                   , help='NHWC parameter, supplied by Experiment')
    self.c                  = self.parser.add_argument('--c'         , type=int  , default=-1                   , help='NHWC parameter, supplied by Experiment')
    self.batch_size         = self.parser.add_argument('--batch_size', type=int  , default=100                  , help='used batch size')
    self.model_type         = self.parser.add_argument('--model_type', type=str  , default='Unknown'            , help='class to load form module "model"')


    if self.data_type == 32: # 32 bit data type
      self.dtype_tf_float, self.dtype_tf_int = tf.float32, tf.int32
      self.dtype_np_float, self.dtype_np_int = np.float32, np.int32
    if self.data_type == 64:                    # 64 bit data type
      self.dtype_tf_float, self.dtype_tf_int = tf.float64, tf.int64
      self.dtype_np_float, self.dtype_np_int = np.float64, np.int64

    self.checkpoint_manager = Checkpoint_Manager(**kwargs)


  @abstractmethod
  def build(self, **kwargs)                  : pass
  def compile(self, **kwargs)                : pass # for external use connect all layers and compile them

  # train on a mini-batch
  @abstractmethod
  def train_step(self, xs, ys=None, **kwargs): pass

  # test on a whole test set
  @abstractmethod
  def test(self, test_iterator, **kwargs)    : pass

  # export ALL of a model's TF variables for TF loading and saving a model.
  # So, the internal state of a model should be defined by its kwargs and its internal tf.Variables's
  def get_model_variables_store_load(self, **kwargs) : return {}


  def load(self, **kwargs):
    ''' try to load an existing checkpoint, return True if checkoint exists else False '''
    log.info(f'try to load checkpoint for model {self.model_type}')
    model_variables = self.get_model_variables_store_load()
    check_point     = self.checkpoint_manager.load_checkpoint(**kwargs, variables=model_variables)
    log.info(f'start at task {check_point}')
    return check_point


  def save(self, **kwargs):
    ''' create a checkpoint for the current model state '''
    model_variables = self.get_model_variables_store_load()
    self.checkpoint_manager.save_checkpoint(**kwargs, variables=model_variables)




