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
from DCGMM.parsers import Kwarg_Parser

class Regularizer():

  def __init__(self, **kwargs):
    prefix      = kwargs.get('prefix', 'Unknown')
    self.name   = f'Regularizer_{prefix}'

    self.parser = Kwarg_Parser(**kwargs)


  def add(self, loss)                : pass
  def set(self, eps=None, sigma=None): pass
  def _check(self)                   : pass


  def check_limit(self):
    self._check()


  def __str__(self):
    return self.__class__.__name__

  def _send(self): return [] # stub to return internal values (for visualization)


class Regularizer_Method():
  DOUBLE_EXP       = 'DoubleExp'
  SINGLE_EXP       = 'SingleExp'
  SINGLE_EXP_BATCH = 'SingleExp_Batch'
  TIME_DECAY       = 'TimeDecayReg'
