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

import math
import numpy      as np
import tensorflow as tf
from .            import Layer
from DCGMM.metric import Metric
from DCGMM.utils  import log

class LOSS_FUNCTION:
  SCE = 'softmax_cross_entropy'
  MSE = 'mean_squared_error'

class Linear_Classifier_Layer(Layer):
  ''' a linear classifier layer '''

  def __init__(self, input=None, **kwargs):
    Layer.__init__(self, input, **kwargs)

    self.name                = self.parser.add_argument('--name'               , type=str  , default=f'{self.prefix}linear', help='name of the gmm layer')
    self.batch_size          = self.parser.add_argument('--batch_size'         , type=int  , default=100                   , help='size of mini-batches we feed from train dataSet.')
    self.num_classes         = self.parser.add_argument('--num_classes'        , type=int  , default=10                    , help='number of output classes')
    self.epsC                = self.parser.add_argument('--epsC'               , type=float, default=0.05                  , help='learn rate') 
    self.epsC                = self.parser.add_argument('--regEps'             , type=float, default=self.epsC             , help='learn rate') # for compatibility of bash files
    self.return_loss         = self.parser.add_argument('--return_loss'        , type=str  , default='loss'                , help='the name of the returned loss tensor')
    self.lambda_W            = self.parser.add_argument('--lambda_W'           , type=float, default=1.0                   , help='adaption factor for Ws')
    self.lambda_b            = self.parser.add_argument('--lambda_b'           , type=float, default=1.0                   , help='adaption factor for bs')
    self.loss_function       = self.parser.add_argument('--loss_function'      , type=str  , default=LOSS_FUNCTION.SCE     , help='the used loss function ["MSE" (Mean Squared Error), "SCE" (Softmax Cross Entropy)]')
    self.sampling_batch_size = self.parser.add_argument('--sampling_batch_size', type=int  , default=100                   , help='sampling batch size')
    self.loss_factor         = self.parser.add_argument('--loss_factor'        , type=float, default=1.                    , help='factor for multiplying resulting layer loss')
    # evaluation metrics
    metrics                  = ['accuracy_score'] + ['loss']
    self.metrics             = self.parser.add_argument('--metrics'            , type=str  , default=metrics               , help='the evaluations metrics')
    self.metric              = Metric(self)

    self.input_shape         = self.prev.get_shape()

    self.channels_in         = np.prod(self.input_shape[1:])
    self.channels_out        = self.num_classes
    log.debug(f'{self.name} input shape {self.input_shape}, sampling_bs={self.sampling_batch_size}')



  def get_shape(self):
    return self.batch_size, self.channels_out


  def get_layer_variables(self, **kwargs):
    tmp = {f'{self.prefix}W': self.W, f'{self.prefix}b': self.b}
    tmp.update({f'extraW_{i}': W for i, (W, _) in enumerate(self.extra_input_variables)})
    tmp.update({f'extrab_{i}': b for i, (_, b) in enumerate(self.extra_input_variables)})
    return tmp


  def is_trainable(self):
    return True


  def compile(self):
    W_shape                    = (self.channels_in, self.channels_out)
    b_shape                    = [self.channels_out]

    init_W                     = tf.initializers.TruncatedNormal(stddev=1. / math.sqrt(self.channels_in))(W_shape)

    self.W                     = self.variable(initial_value=init_W, name='weight', shape=W_shape)
    self.b                     = self.variable(np.zeros(b_shape)   , name='bias'  , shape=b_shape)

    self.extra_input_variables = list()

    # constants to change the adaption rate by SGD step (Ws, bs)
    self.lambda_W_factor       = self.variable(self.lambda_W, name='lambda_W', shape=[])
    self.lambda_b_factor       = self.variable(self.lambda_b, name='lambda_b', shape=[])


  @tf.function(autograph=False)
  def forward(self, input_tensor, extra_inputs=[]):
    tensor_flattened = tf.reshape(input_tensor, (-1, self.channels_in))
    logits           = tf.nn.bias_add(tf.matmul(tensor_flattened, self.W), self.b)
    for (extra_W, extra_b), extra_input_tensor in zip(self.extra_input_variables, extra_inputs):
      tensor_flattened = tf.reshape(extra_input_tensor, (self.batch_size, -1))
      logits           = logits + tf.nn.bias_add(tf.matmul(tensor_flattened, extra_W), extra_b)
    return logits


  def loss(self, logits, **kwargs):
    ''' classifier operator (supervised) train a linear classifier (output: predicted class label) '''
    labels = kwargs.get('ys')
    return self.graph_loss(logits, labels)


  #@tf.function(autograph=False)
  def graph_loss(self, logits, labels):
    if self.loss_function == LOSS_FUNCTION.SCE: loss = -tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    if self.loss_function == LOSS_FUNCTION.MSE: loss = -tf.reduce_sum((logits - labels) ** 2, axis=1)
    return loss


  def evaluate(self, logits, **kwargs):
    ''' evaluation on a mini-batch '''
    y_true_onehot = kwargs.get('ys')
    y_true        = tf.argmax(y_true_onehot, axis=1)
    y_pred        = tf.argmax(logits, axis=1)                                      # class value e.g. [0, 1, 3, ...]
    #y_pred_onehot = tf.one_hot(tf.cast(y_pred, dtype=tf.int32), self.num_classes) # class value in one-hot format

    # accuracy is calculated automatically by y_true and y_pred by Metric object
    accuracy       = tf.equal(y_true, y_pred)
    accuracy       = tf.cast(accuracy, self.dtype_tf_float)
    accuracy       = tf.reduce_mean(accuracy)
    result         = {
      'accuracy': accuracy,
      'y_true'  : y_true  ,
      'y_pred'  : y_pred  ,
      }

    return result


  def update_with_grad(self, grads):
    self.W.assign_add(self.lambda_W_factor * self.epsC * grads[f'{self.prefix}W'])
    self.b.assign_add(self.lambda_b_factor * self.epsC * grads[f'{self.prefix}b'])

    for i, (W, b) in enumerate(self.extra_input_variables):
      W.assign_add(self.lambda_W_factor * self.epsC * grads[f'extraW_{i}'])
      b.assign_add(self.lambda_b_factor * self.epsC * grads[f'extrab_{i}'])


  def backwards(self, topdown=None, **kwargs):
    ''' topdown is a 2D tensor_like of shape [sampling_batch_size,num_classes] in in one-hot! '''
    input_shape    = self.prev.get_shape()
    input_shape[0] = self.sampling_batch_size

    if topdown is None:
      return tf.ones(self.input_shape)

    # logits are created as: L = WX + b --> so X = WinvL - b. we approximate inv(W) by W.T
    sampling_op = tf.cast(tf.matmul(topdown - tf.expand_dims(self.b, 0), tf.transpose(self.W)), self.dtype_tf_float)
    sampling_op = tf.reshape(sampling_op - tf.reduce_min(sampling_op, axis=1, keepdims=True), input_shape)
    return sampling_op


  def post_test_step(self, results, xs, ys=None, **kwargs):
    y_pred        = results.get('y_pred')
    y_true        = results.get('y_true')
    loss          = results.get('loss')

    metric_values = self.metric.eval(
      dict    = True  , # return a dictionary with metric values
      y_true  = y_true,
      y_pred  = y_pred,
      loss    = loss * self.loss_factor,
      special = {
        'accuracy_score': dict(normalize=True),
        }
      )
    return metric_values


  def share_variables(self, *args, **kwargs):
    ''' label the previous layer prototyps '''
    data_dict = dict(proto_labels=list())
    def label_mus():
      if not self.prev.is_layer_type('GMM')          : return
      if self.prev.h_out != 1 or self.prev.w_out != 1: return # can not feed patches respectively add labels to slices
      n                         = int(math.sqrt(self.prev.c_in // 1)) # TODO: set to 3 if color image
      mus                       = self.prev.mus
      reshape_mus               = tf.reshape(mus, [self.prev.K, 1, 1, n *  n]) # convert mus to images to use as input
      responsibilities          = self.prev.forward(reshape_mus)
      logits                    = self.forward(responsibilities)
      y_pred                    = tf.argmax(logits, axis=1) # class value e.g. [0, 1, 3, ...]
      labels                    = tf.reshape(y_pred, [self.prev.n, self.prev.n])
      data_dict['proto_labels'] = labels.numpy()
    label_mus()

    return data_dict

