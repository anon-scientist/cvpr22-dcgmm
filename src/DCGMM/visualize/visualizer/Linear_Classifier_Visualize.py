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
from visualize.visualizer.Network_Queue_Visualizer import Network_Queue_Visualizer

if __name__ == '__main__':
  vis = Network_Queue_Visualizer(
    #server_address = '192.168.41.10',
    server_port    = 443           ,
    name           = 'Linear Layer' ,
    from_layers=[
      ['environment', 'mean_loss'],
      ],
    )
  vis.go() # blocking
    

