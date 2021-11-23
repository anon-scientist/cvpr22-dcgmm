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
import tensorflow as tf
from DCGMM.utils        import log
from DCGMM.parsers      import Kwarg_Parser

class Checkpoint_Manager():

  def __init__(self, **kwargs):
    parser         = Kwarg_Parser(**kwargs)

    self.model_type = parser.add_argument('--model_type', type=str  , default='Unknown'       , help='class to load form module "model"')
    self.exp_id     = parser.add_argument('--exp_id'    , type=str  , default='0'             , help='unique experiment id (for experiment evaluation)')
    self.ckpt_dir   = parser.add_argument('--ckpt_dir'  , type=str  , default='./checkpoints/', help='directory for checkpoint files')
    self.load_task  = parser.add_argument('--load_task' , type=int  , default=0               , help='<task> load a specified task checkpoint (0 = do not load checkpoint, load=<exp_id>_D<task>)')
    self.save_All   = parser.add_argument('--save_All'  , type=bool , default=False           , help='if specified, model is saved after last training iteration for each task (store=<exp_id>_D<task>)')

    self.tasks_checkpoints = [ int(task.split('_', 1)[1][1:]) for task, save in kwargs.items() if task.startswith('save_D') and save ] # e.g., [1, 3]
    self.filename          = os.path.join(self.ckpt_dir, f'{self.exp_id}-{self.model_type.lower()}-{{}}.ckpt')


  def load_checkpoint(self, **kwargs):
    ''' load a model configuration via checkpoint manager '''
    if not kwargs.get('task'):
      if not hasattr(self, 'load_task'): return 0
      if self.load_task <= 0           : return 0
    log.debug(f'loading {self.load_task}')
    try:
      task = kwargs.get('task')
      if (task): self.load_task = task
      checkpoint_filename = self.filename.format(self.load_task)
      model_vars          = kwargs.get('variables')
      if not hasattr(self, 'saver'): self.saver = tf.train.Checkpoint(**model_vars)
      self.saver.restore(checkpoint_filename)
      log.info(f'restore model from {checkpoint_filename} start at task {self.load_task}')
    except Exception as ex:
      log.error(f'problem at load time: {ex}')
      self.load_task = 0
      #raise ex
    return self.load_task


  def save_checkpoint(self, **kwargs):
    ''' save current session state to disk '''
    save_All     = getattr(self, 'save_All', None)
    task         = kwargs.get('task')
    if save_All is None and len(self.tasks_checkpoints) == 0 and task == None: return

    try   : current_task = kwargs.get('current_task', None) + 1
    except: current_task = task

    if save_All or current_task in self.tasks_checkpoints or task:
      model_vars          = kwargs.get('variables')
      self.saver          = tf.train.Checkpoint(**model_vars) if not hasattr(self, 'saver') else self.saver # to store and restore models # defer_build=True
      checkpoint_filename = self.filename.format(current_task)
      self.saver.write(checkpoint_filename)
      log.info(f'create checkpoint for session to {checkpoint_filename}')
      return

    log.debug(f'do not create checkpoint for task {current_task}')

