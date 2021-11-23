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

class Kwarg_Parser(object):
  ''' A simple parser for kwargs. Behaves similarly to the standard argparse module.
    Check the parameter added with the "add_argument" method if they are present in kwargs.
    If the parser has a prefix, it will first search for the parameter with prefix then without.
    Parameter priority: 1. command line (if present) 2. kwargs 3. default value
  '''

  def __init__(self, prefix='', external_arguments=None, verbose=False, **kwargs):
    ''' init Kwarg_Parser
    @param prefix: search for parameters with the prefix (e.g., prefix="L3" parameter="--K" search for parameter "--L3_K")
    @param command_line_arguments: dictionary of external (e.g., command line) parameters
    '''
    self.kwargs  = kwargs
    self.verbose = verbose
    if external_arguments: self.kwargs.update(external_arguments) # overwrite kwargs with command line parameters
    self.prefix   = prefix
    self.help_str = ''


  def convert(self, op, obj):
    ''' applies op to convert the type of object. If object is a list, conversion is element-wise '''
    if isinstance(obj, list): return list(map(op, obj))
    else                    : return op(obj)


  def get_all_parameters(self):
    ''' return all collected arguments as dict '''
    return self.kwargs


  def add_argument(self, arg_name, type=str, default=None, required=False, help='', choices=None, prefix=True, **kwargs):
    ''' assumes 1st 2 chars are -- and ignores them TODO: n1 documentation '''
    if not arg_name.startswith('--'): raise Exception(f'argument ({arg_name}) does not start with "--" ')

    arg_name          = arg_name[2:]                                         # remove --
    param_value_prio1 = self.kwargs.get(f'{self.prefix}{arg_name}')          # get value from kwargs with prefix
    param_value_prio2 = self.kwargs.get(arg_name)                            # value from arg without prefix
    param_value       = param_value_prio1
    if param_value is None: param_value = param_value_prio2

    if self.verbose == True:
      #raise Exception('unallowed "print" Exception')
      #print(f"{self.prefix}: looking for {arg_name}, found {param_value_prio1} as priority 1 and {param_value_prio2} as prio2 --> kept {param_value}")
      pass

    if param_value is None and required    : raise Exception(f'Invalid kwargs: {arg_name} missing!') # if required arg is missing
    if param_value is None and not required: param_value = default                                   # if arg is missing use default value
    else                                   : param_value = self.convert(type, param_value)           # if arg is given apply type convert function

    if choices and param_value not in choices: raise Exception(f'Invalid choice: {arg_name}={param_value} not in {choices}') # should choices be possible, then check if is included.
    if arg_name not in self.kwargs: self.kwargs[arg_name] = param_value                                                      # collect all parameter in self.kwargs even the parameter is not given

    self.help_str += f'\n{help}'
    return param_value

