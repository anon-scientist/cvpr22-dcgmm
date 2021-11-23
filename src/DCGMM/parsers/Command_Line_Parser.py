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
import sys

class Command_Line_Parser(object):
  ''' read in command line arguemnts '''

  def __init__(self, **kwargs):
    self.command_line_parameter_list = sys.argv[1:]

  def parse_args(self):
    ''' AGP '''
    unknown     = self.command_line_parameter_list
    unknownDict = {}
    pName = ""
    for item in unknown:
      if '--' in item:
        # check of valuelist oif previous param is one-elem
        if pName != "":
          lst = unknownDict[pName]
          if len(lst) == 1:
            unknownDict[pName] = lst[0]
        pName = item[2:]
        unknownDict[pName] = []
      else:
        unknownDict[pName].append(item)

    if pName != "":
      lst = unknownDict[pName]
      if len(lst) == 1: unknownDict[pName] = lst[0]
    return unknownDict
