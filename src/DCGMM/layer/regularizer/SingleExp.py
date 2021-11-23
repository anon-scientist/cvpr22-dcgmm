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
from . import Regularizer
from DCGMM.utils import log


class SingleExp(Regularizer):

  def __init__(self, tfEps, tfSomSigma, eps0, somSigma0, somSigmaInf, epsInf, **kwargs):
    super().__init__(**kwargs)

    self.tf_eps                = tfEps
    self.tf_som_sigma          = tfSomSigma
    self.eps0                  = eps0
    self.somSigma0             = somSigma0
    self.somSigmaInf           = somSigmaInf
    self.epsInf                = epsInf

    self.alpha                 = self.parser.add_argument('--alpha'       , type=float, default=self.epsInf     , help = 'reaction speed (higher is slower)' )
    self.gamma                 = self.parser.add_argument('--gamma'       , type=float, default=0.9             , help = 'reduction factor of somSigma' )
    self.delta                 = self.parser.add_argument('--delta'       , type=float, default=0.05            , help = 'stationarity detection threshold' )
    self.reset_sigma           = self.parser.add_argument('--reset_sigma' , type=float, default=self.somSigmaInf, help = 'reset value for sigma') # default, reset to somSigmaInf, even if not yet completely reduced
    self.reset_eps             = self.parser.add_argument('--reset_eps'   , type=float, default=self.epsInf     , help = 'reset value for eps')   # default, reset to

    #log.debug(f'Delta = {self.delta}')

    self.avgLong               = 0.0
    self.l0                    = 0
    self.lastAvg               = 0
    self.iteration             = 0
    self.W                     = int(1 / self.alpha)
    self.limit                 = None
    self.currentSigma          = self.somSigma0


  def __str__(self):
    s = ''
    s += f'Regularizer: {self.prefix}{self.name}\n'
    s += f' delta      : {self.delta}\n'
    s += f' gamma      : {self.gamma}\n'
    s += f' eps0       : {self.eps0}\n'
    s += f' epsInf     : {self.epsInf}\n'
    s += f' somSigma0  : {self.somSigma0}\n'
    s += f' sigmaInf   : {self.somSigmaInf}\n'
    s += f' reset_sigma: {self.reset_sigma}\n'
    s += f' reset_eps  : {self.reset_eps}\n'
    s += f' alpha      : {self.alpha}'
    return s


  def add(self, loss):
    itInLoop        = self.iteration % self.W
    self.losses     = loss
    if itInLoop == 0: self.avgLong = 0.0

    self.avgLong   *= itInLoop / (itInLoop + 1.)
    self.avgLong   += loss / (itInLoop + 1.)

    self.iteration += 1
    return self


  def set(self, eps=None, sigma=None):
    ''' reset the regularizer '''
    log.debug(f'{self.name} set eps={eps} and sigma={sigma} AT {self.avgLong} AND {self.iteration}')
    reset_eps   = eps   if eps   else self.reset_eps
    reset_sigma = sigma if sigma else self.reset_sigma

    reset_eps   = reset_eps   if reset_eps   > self.epsInf   else self.epsInf
    reset_sigma = reset_sigma if reset_sigma > self.somSigmaInf else self.somSigmaInf

    if reset_sigma > self.somSigmaInf: log.info(f'regularize {self.name} set sigma={reset_sigma:1.5f} and eps={reset_eps:1.5f} at iteration {self.iteration}')

    self.tf_som_sigma.assign(reset_sigma)
    self.tf_eps.assign(reset_eps)


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
    self.limit  = limit

    if limit > -2 * self.delta and limit < self.delta: # if energy does not increase sufficiently --> reduce!
      currentEps, self.currentSigma = [self.tf_eps.numpy(), self.tf_som_sigma.numpy()]
      currentEps        *= self.gamma
      self.currentSigma *= self.gamma
      self.set(eps=currentEps, sigma=self.currentSigma)

    self.lastAvg = self.avgLong # update lastAvg for next event


  def _send(self):
    avg_long  = self.avgLong if self.avgLong else self.losses
    avg_short = self.lastAvg if self.lastAvg else self.losses
    avg_diff  = self.l0      if self.l0      else self.losses
    long_time = self.limit   if self.limit   else self.losses

    return [
      ('avgLong', avg_long) , # add samples for visualization
      ('lastAvg', avg_short), # add samples for visualization
      ('l0'     , avg_diff) , # add samples for visualization
      ('limit'  , long_time), # add samples for visualization
      ]
