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
import os
import time
import json
from collections       import defaultdict
from DCGMM.utils       import log
from DCGMM.parsers     import Kwarg_Parser
import numpy     as np

class Logging(object):

  def __init__(self, **kwargs):
    ''' structured and uniform logging (json format) '''
    parser             = Kwarg_Parser(**kwargs)

    self.exp_id        = parser.add_argument     ('--exp_id'       , type=str  , default='0'      , help='unique experiment id (for experiment evaluation)')
    self.tmp_dir       = parser.add_argument     ('--tmp_dir'      , type=str  , default='.'      , help='directory for output files (for experiment evaluation)')
    self.model_type    = parser.add_argument     ('--model_type'   , type=str  , default='Unknown', help='class to load form module "model"')
    self.results_dir   = parser.add_argument     ('--results_dir'  , type=str  , default='./results'  , help='set the default directory to search for dataset files')

    filename           = f'{self.exp_id}_log.json'
    self.output_file   = os.path.join(self.results_dir, filename)
    self.log           = {
      'parameters' : dict()           , # <name>: <value>, ...
      'eval'       : defaultdict(list), # <metric>_<taskname>_<layer>: [<task>, <iteration>, <value>], ... # task -1=DAll, 0=DNow, x=Dx
      'created'    : time.asctime()   , # <timestamp>
      }

    for k,v in kwargs.items():
      self.add_parameter(k, v)


  def _is_jsonable(self, k, v):
    def check(x, name):
      try:
        json.dumps(x)
        if isinstance(v, list) and len(v) == 0: return False # remove empty lists that "could" be filled with not serializable values
        return True
      except Exception as ex:
        log.debug(f'could not serialize {name} {k}: {v} because {ex}')
        return False
    return check(k, 'key') and check(v, 'value')


  def add_parameter(self, k, v):
    if not self._is_jsonable(k, v): return
    self.log['parameters'][k] = v


  def add_eval(self, k, v):
    if not self._is_jsonable(k, v): return
    self.log['eval'][k].append(v)


  def add_eval_name_value(self,
      metricname = 'nometric'   ,
      taskname   = 'notask'     ,
      layername  = 'nolayer'    ,
      task       = -2           ,
      iteration  = -1           ,
      value      = -1           ,
    ):
    k = f'{metricname}_{taskname}_{layername}'
    v = [task, iteration, value]
    try                   : self.add_eval(k, v)
    except Exception as ex: print(metricname, taskname, layername, task, iteration, value, ex)


  def add_eval_combination(self, keys, values):
    ''' create a key string of keys and value list of values '''
    k = '_'.join(keys)
    vals = list()
    for v in values: vals.append(float(v) if isinstance(v, np.floating) else v) # detect and substitute numpy floats with python floats
    self.add_eval(k, vals)


  def write_to_file(self):
    log.info(f'write logging to {self.output_file}')
    with open(self.output_file, 'w') as file:
      json.dump(self.log, file)
      # json.load(open(self.output_file, 'r')) # test load file

