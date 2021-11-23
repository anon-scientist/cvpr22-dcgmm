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
from .        import Layer

class Input_Layer(Layer):

  def __init__(self, input=None, **kwargs):
    Layer.__init__(self, input=input, **kwargs)

    self.name    = self.parser.add_argument('--name'      , type=str, default=f'{self.prefix}input', help='name of the input layer')
    self.N       = self.parser.add_argument('--batch_size', type=int , default=100                          )
    self.h       = self.parser.add_argument('--h'         , type=int , default=-1                           )
    self.w       = self.parser.add_argument('--w'         , type=int , default=-1                           )
    self.c       = self.parser.add_argument('--c'         , type=int , default=-1                           )
    self.flatten = self.parser.add_argument('--flatten'   , type=bool, default=False                        )


  def get_shape(self):
    return [self.N, self.h, self.w, self.c]

