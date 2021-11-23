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
from . import Regularizer
import tensorflow as tf


class TimeDecayReg(Regularizer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.max_iter      = kwargs.get('max_iterations', None)
    self.t0Frac        = self.add_reg_param('t0Frac'  , type=float, default=0.3, help='start reduction point for time_decay regularizer in %% of max_iteration of task for eps and sigma (start eps0)')
    self.tInfFrac      = self.add_reg_param('tInfFrac', type=float, default=0.8, help='end reduction point for time_decay regularizer in %% of max_iteration of task for eps and sigma (end epsInf and somSigmaInf)')

    self.eps0          = kwargs.get('eps0'       )
    self.epsInf        = kwargs.get('epsInf'     )
    self.tfEps         = kwargs.get('tfEps'      )
    self.tfSomSigma    = kwargs.get('tfSomSigma' )
    self.somSigma0     = kwargs.get('somSigma0'  )
    self.somSigmaInf   = kwargs.get('somSigmaInf')

    self.t0            = self.t0Frac   * self.max_iter
    self.t1            = self.tInfFrac * self.max_iter


    self.kappaEps      = math.log(self.eps0 / self.epsInf) / (self.t1 - self.t0)
    self.kappaSigma    = math.log(self.somSigma0 / self.somSigmaInf) / (self.t1 - self.t0)

    # define Tf operators only once so there are no memory leaks
    self.epsPH         = tf.placeholder(shape=[], dtype=tf.float64)
    self.sigmaPH       = tf.placeholder(shape=[], dtype=tf.float64)
    self.updateEpsOp   = tf.assign(self.tfEps, self.epsPH)
    self.updateSigmaOp = tf.assign(self.tfSomSigma, self.sigmaPH)

    self.iteration     = 0


  def extend_parameters(self, parser):
    parser.add_argument(f'--L{self.layer_id}_t0Frac'  , type=float, default=0.3 , help='start reaction')
    parser.add_argument(f'--L{self.layer_id}_tInfFrac', type=float, default=0.8 , help='end reaction')


  def add(self, *args, **kwargs):
    self.iteration += 1
    return self


  def set(self, eps=None, somSigma=None):
    ''' set the regularizer state'''
    self.sess.run(self.updateSigmaOp, feed_dict={self.sigmaPH: somSigma})
    self.sess.run(self.updateEpsOp, feed_dict={self.epsPH: eps})

  def _check(self):
    if  self.iteration < self.t0 or self.iteration > self.t1: return

    self.set(
      eps      = self.eps0      * math.exp(-self.kappaEps   * (self.iteration - self.t0)),
      somSigma  = self.somSigma0 * math.exp(-self.kappaSigma * (self.iteration - self.t0)),
      )
