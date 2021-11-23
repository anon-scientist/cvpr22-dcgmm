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
import tensorflow             as tf

from .regularizer.Regularizer   import Regularizer_Method as RM
from layer                      import GMM_Layer
from metric                     import Metric
from importlib                  import import_module

class Mode:
  DIAG   = 'diag'
  FULL   = 'full'
  MFA    = 'mfa'

class Energy:
  LOGLIK     = 'loglik'
  MC         = 'mc'
  LOGLIK_ANN = 'loglik_ann'


class GMM_MFA_Layer(GMM_Layer):
  ''' a GMM-MFA layer '''

  def __init__(self, input=None, **kwargs):
    ''' GMM-MFA Layer
    @param input_: input tensor of the previous layer
    @param kwargs: parameter dictionary with all parameters (all layers),
           layer specific parameters are in self.layer_params (extracted in base class)
    '''
    GMM_Layer.__init__(self, input, **kwargs)
    self.name                  = self.add_layer_param('name'    , type=str  , default=f'{self.prefix}mfa', help='name of the gmm layer')
    #------------------------------------------------------------ MFA PARAMETERS
    self.lambda_D              = self.add_layer_param('lambda_D', type=float, default=1.                 , help='factor for the Ds in MFA')



  def get_shape(self):
    return [self.batch_size, self.h_out, self.w_out, self.c_out]


  def get_layer_variables(self):
    ''' This saves all trainable parameters of the current layer into a dictionary to return '''
    train_variables = super.get_layer_variables()
    train_variables.update({f'{self.prefix}D' : self.D})
    return train_variables


  def is_trainable(self): return True


  def evaluate(self, logits, **kwargs):
    return {'log_likelihood': self.return_loss, 'log_likelihoods': self.loglikelihood_full}


  def _init_tf_variables(self):
    super._init_tf_variables()

    self.lambda_D_factor     = self.variable(1.               , shape=[],  name='lambda_D')
    D_shape                  = [1, self.h_out, self.w_out, self.K, self.c_in]

    if self.convMode:
      D_shape[1]      = D_shape[2]      = 1

    self.D                   = tf.constant(1.0)


  def update_with_grad(self, grads):
    self.pis.assign_add(        self.lambda_pi_factor    * self.tfEps * grads[self.prefix + 'pis']   )
    self.mus.assign_add(        self.lambda_mu_factor    * self.tfEps * grads[self.prefix + 'mus']   )
    self.sigmas.assign_add(     self.lambda_sigma_factor * self.tfEps * grads[self.prefix + 'sigmas'])
    self.lambda_D_factor.assign(self.lambda_D_factor     * self.tfEps * grads[self.prefix + 'D']     )

    if self.mode == Mode.DIAG: # sigma clipping for diag!
      sigma_limit = math.sqrt(self.sigmaUpperBound)
      self.sigmas.assign(tf.clip_by_value(self.sigmas, -sigma_limit, sigma_limit))

    last_log_likelihood = tf.reduce_mean(self.loglikelihood_full) # batch/patch mean of log-likelihood
    self.reg.add(last_log_likelihood).check_limit()


  def reset_layer(self, **kwargs):
    super.reset_layer(**kwargs)
    reset_factor   = kwargs.get('reset_factor')

    if reset_factor == 1.0: # full reset
      self.D.assign(tf.constant(1.0))
