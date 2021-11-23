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


class SingleExp_Batch(Regularizer):
  ''' a batch varient of SingleExp regularizer '''

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    kwargs['_name']     = kwargs.get('name', f'{self.prefix}SingleExpBatch')

    self.eps0           = kwargs.get('eps0'       )
    self.epsInf         = kwargs.get('epsInf'     )
    self.somSigma0      = kwargs.get('somSigma0'  )
    self.sigmaInf       = kwargs.get('somSigmaInf')
    self.tfEps          = kwargs.get('tfEps'      )
    #self.tfSomSigma     = kwargs.get('tfSomSigma' )
    self.sigmaPH        = kwargs.get('tfSomSigma' )
    self.sigmaPH

    self.delta          = self.add_reg_param('delta'      , type=float, default=0.01         , help='reaction speed')
    self.gamma          = self.add_reg_param('gamma'      , type=float, default=0.9          , help='reduction factor of single_exp regularizer for eps and sigma (per step)')
    self.reset_sigma    = self.add_reg_param('reset_sigma', type=float, default=self.sigmaInf, help='reset value for sigma') # default, reset to somSigmaInf, even if not yet completely reduced
    self.reset_eps      = self.add_reg_param('reset_eps'  , type=float, default=self.epsInf  , help='reset value for eps')   # default, reset to

    self.alpha          = self.epsInf
    self.avgLong        = 0.0
    self.l0             = 0
    self.lastAvg        = 0
    self.iteration      = 0
    self.W              = int(1 / self.alpha)

    self.sess           = tf.compat.v1.get_default_session() # define TF operators only once so there are no memory leaks
    self.dtype_tf_float = kwargs.get('dtype_tf_float', tf.float32)
    self.epsPH          = self.placeholder(name='epsPH'  , shape=[], input=self.const(self.eps0     , dtype=self.dtype_tf_float))
    self.sigmaPH        = self.placeholder(name='sigmaPH', shape=[], input=self.const(self.somSigma0, dtype=self.dtype_tf_float))

    self.updateEpsOp    = tf.assign(self.tfEps     , self.epsPH  )
    #self.updateSigmaOp  = tf.assign(self.tfSomSigma, self.sigmaPH)

    self.current_somSigma =self.somSigma0


  def __str__(self):
    s = ''
    s += f'Regularizer: {self.prefix}{self._name}\n'
    s += f' delta      : {self.delta}\n'
    s += f' gamma      : {self.gamma}\n'
    s += f' eps0       : {self.eps0}\n'
    s += f' epsInf     : {self.epsInf}\n'
    s += f' somSigma0  : {self.somSigma0}\n'
    s += f' sigmaInf   : {self.sigmaInf}\n'
    s += f' reset_sigma: {self.reset_sigma}\n'
    s += f' reset_eps  : {self.reset_eps}\n'
    s += f' alpha      : {self.alpha}'
    return s


  def add(self, loss):
    loss = loss.mean()
    itInLoop        = self.iteration % self.W
    if itInLoop == 0: self.avgLong = 0.0

    self.avgLong   *= itInLoop / (itInLoop + 1.)
    self.avgLong   += loss / (itInLoop + 1.)
    self.iteration += 1
    return self


  def set(self, eps=None, sigma=None):
    ''' reset the regularizer '''
    log.debug(f'{self._name} set eps={eps} and sigma={sigma} AT {self.avgLong} AND {self.iteration}')
    reset_eps   = eps   if eps   else self.reset_eps
    reset_sigma = sigma if sigma else self.reset_sigma

    reset_eps   = reset_eps   if reset_eps   > self.epsInf   else self.epsInf
    reset_sigma = reset_sigma if reset_sigma > self.sigmaInf else self.sigmaInf

    #self.sess.run(self.updateSigmaOp, feed_dict={self.sigmaPH: reset_sigma})
    self.current_somSigma = reset_sigma
    self.sess.run(self.updateEpsOp  , feed_dict={self.epsPH  : reset_eps})
    #print('reduce somSigma to', reset_sigma)


  def _check(self):
    # do nothing during the first period just memorize initial loss value as a baseline
    if self.iteration % self.W != self.W - 1: # if we are not at the end of a period, do nothing
      return

    if self.iteration // self.W == 0:         # first event: no lastAvg yet, so set it trivially, no action
      self.l0      = self.avgLong
      self.lastAvg = self.l0
      return

    if self.iteration // self.W == 1:         # second event: lastAvg can be set non-trivially, no action
      self.lastAvg = self.avgLong
      return

    limit       = (self.avgLong - self.lastAvg) / (self.lastAvg - self.l0)

    if limit > -2 * self.delta and limit < self.delta: # if energy does not increase sufficiently --> reduce!
      #currentEps, currentSigma = self.sess.run([self.tfEps, self.tfSomSigma])
      currentEps = self.sess.run(self.tfEps)
      currentEps            *= self.gamma
      self.current_somSigma *= self.gamma
      self.set(eps=currentEps, sigma=self.current_somSigma)

    self.lastAvg = self.avgLong # update lastAvg for next event
