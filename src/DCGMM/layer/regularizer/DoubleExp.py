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
import           tensorflow as tf
from .           import Regularizer
from DCGMM.utils import log


class DoubleExp(Regularizer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    kwargs['_name']           = kwargs.get('name', f'{self.prefix}SingleExp_Adaption')
    self.connection_handler   = kwargs.get('connection_handler')

    self.delta                = self.add_reg_param('delta', type=float, default=0.01, help='reaction speed')
    self.gamma                = self.add_reg_param('gamma', type=float, default=0.9 , help='decreasing factor')
    
    self.avg_long_time_factor = self.add_reg_param('zeta', type=float, default=0.0001, help='long time factor')
    self.avg_long_factor      = self.add_reg_param('alpha', type=float, default=0.005, help='long average factor')
    self.avg_short_factor     = self.add_reg_param('beta' , type=float, default=0.01 , help='short average factor')
    self.avg_diff_factor      = self.add_reg_param('kappa', type=float, default=0.1 , help='diff average factor')
    
    self.wait_iterations      = self.add_reg_param('wait' , type=int  , default=100 , help='iterations to wait before starting regularization') # TODO: same with samples? 
                              
    self.long_time            = None 
    self.avg_long             = None
    self.avg_short            = None
    self.avg_diff             = None
                              
    self.iteration            = 0
    self.wait_again           = 0
                              
    self.sess                 = tf.compat.v1.get_default_session() # define TF operators only once so there are no memory leaks
    self.dtype_tf_float       = kwargs.get('dtype_tf_float', tf.float32)
                              
    self.add                  = self._add_wait   # add order : add_wait, add_init, add
    self._check               = self._check_wait # check oder: check_wait, check_active
    
    self.init_tf_(**kwargs)
  
  
  def init_tf_(self, **kwargs):
    self.eps0           = kwargs.get('eps0'       )
    self.epsInf         = kwargs.get('epsInf'     )
    self.somSigma0      = kwargs.get('somSigma0'  )
    self.sigmaInf       = kwargs.get('somSigmaInf')
    self.tfEps          = kwargs.get('tfEps'      )
    self.tfSomSigma     = kwargs.get('tfSomSigma' )
    self.currentSigma   = self.somSigma0
    
    self.sess           = tf.compat.v1.get_default_session() # define TF operators only once so there are no memory leaks
    self.dtype_tf_float = kwargs.get('dtype_tf_float', tf.float32)
    self.epsPH          = self.placeholder(name='epsPH'  , shape=[], input=self.const(self.eps0     , dtype=self.dtype_tf_float))
    self.sigmaPH        = self.placeholder(name='sigmaPH', shape=[], input=self.const(self.somSigma0, dtype=self.dtype_tf_float))

    self.updateEpsOp    = tf.compat.v1.assign(self.tfEps     , self.epsPH  )
    self.updateSigmaOp  = tf.compat.v1.assign(self.tfSomSigma, self.sigmaPH)
    

  def _add_wait(self, losses):
    ''' do nothing during the first period '''
    self.losses     = losses
    self.iteration += 1
    if self.iteration < self.wait_iterations // 2: return self
    self.long_time  = losses
    self.add        = self._add_init_long
    return self


  def _add_init_long(self, losses):
    ''' initialize both averages same '''
    self.losses     = losses
    self.iteration += 1
    self.avg_long   = losses
    self.add        = self._add_init_short
    return self
  
  def _add_init_short(self, losses):
    self.losses     = losses
    self.iteration += 1
    if self.iteration < self.wait_iterations // 1.5: return self
    self.avg_short  = losses
    self.long_time  = (self.avg_long_time_factor  * losses) + ((1 - self.avg_long_time_factor) * self.long_time)
    self.add        = self._add
    self.avg_diff   = self.avg_long - self.avg_short
    return self


  def _add(self, losses):
    ''' update sliding averages '''
    self.iteration += 1
    self.losses     = losses
    self.avg_long   = (self.avg_long_factor      * losses) + ((1 - self.avg_long_factor)      * self.avg_long )
    self.avg_short  = (self.avg_short_factor     * losses) + ((1 - self.avg_short_factor)     * self.avg_short)
    self.long_time  = (self.avg_long_time_factor * losses) + ((1 - self.avg_long_time_factor) * self.long_time)
    return self

  def _check_wait(self):
    ''' wait with regularization until all wait iterations are over '''
    if self.iteration < self.wait_iterations: return
    self._check = self._check_active 

  def set(self, eps=None, sigma=None):
    ''' reset the regularizer '''
    
    reset_eps   = eps   if eps   else self.reset_eps
    reset_sigma = sigma if sigma else self.reset_sigma

    reset_eps   = reset_eps   if reset_eps   > self.epsInf   else self.epsInf
    reset_sigma = reset_sigma if reset_sigma > self.sigmaInf else self.sigmaInf
    
    if reset_sigma > self.sigmaInf: log.info(f'regularize set sigma={reset_sigma:1.5f} and eps={reset_eps:1.5f} at iteration {self.iteration}' )
    self.sess.run(self.updateSigmaOp, feed_dict={self.sigmaPH: reset_sigma})
    self.sess.run(self.updateEpsOp  , feed_dict={self.epsPH  : reset_eps})

  def _check_active(self):
    if self.wait_again > 0: 
      self.wait_again -= 1
      return     
    
    diff          = self.avg_long - self.avg_short
    self.avg_diff = self.avg_diff_factor * diff + (1 - self.avg_diff_factor) * self.avg_diff
    
    #print(self.avg_diff ** 2)
    if self.avg_diff ** 2 < 10.0:
      #if self.avg_diff < 0: print('reduce', self.avg_diff)
      #else: print('increase', self.avg_diff)
      currentEps, currentSigma = self.sess.run([self.tfEps, self.tfSomSigma])
      currentEps              *= self.gamma if self.avg_diff < 0 else (1. - self.gamma) + 1. 
      currentSigma            *= self.gamma if self.avg_diff < 0 else (1. - self.gamma * 5) + 1.
      self.currentSigma        = currentSigma
      self.set(eps=currentEps, sigma=currentSigma)

      self.wait_again = self.wait_iterations  # minimum waiting iterations
    
  def _send(self):
    avg_long  = self.avg_long  if self.avg_long  else self.losses  
    avg_short = self.avg_short if self.avg_short else self.losses
    avg_diff  = self.avg_diff  if self.avg_diff  else self.losses
    long_time = self.long_time if self.long_time else self.losses
    
    return [
      ('reg_long' , avg_long) ,        # add samples for visualization
      ('reg_short', avg_short),        # add samples for visualization
      ('reg_diff' , avg_diff) ,        # add samples for visualization
      ('reg_time' , long_time) ,       # add samples for visualization
      ]
