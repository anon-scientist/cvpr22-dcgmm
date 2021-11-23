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
import matplotlib.pyplot as plt
from visualize import Stream_Client
import numpy as np
import json

class Visualizer(Stream_Client):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.name              = kwargs.get('name'                         )
    self.plot_only         = kwargs.get('plot_only'        , True      )
    self.grid              = kwargs.get('grid'             , True      )
    self.enabled           = kwargs.get('enabled'          , True      )
    self.output_dir        = kwargs.get('output_dir'       , './plots/')
    self.shutdown          = False
    self.config            = kwargs

    if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

    self.connect()
    self._init_visualization()
    self.start()

  def init_plot(self)        : pass
  def update_plot(self, data): pass

  def _add_grid(self, data, val=1.0):
    if self.grid: # add grid lines
      for i in range(self.x_plots + 1):
        data = np.insert(data, i * self.w + i, val, axis=1)
        data = np.insert(data, i * self.h + i, val, axis=0)
    return data


  def send_and_receive_config(self):
    self.socket.send(json.dumps(self.config).encode())
    config_dict          = self.socket.recv(1024).decode()
    print('get config', config_dict)
    self.received_config = json.loads(config_dict)


  def process(self, data):
    ''' start the preprocessing step for the next incoming block (called by base class)

    @param data: input data from stream server (np.array)
    @return: None (never stop the processing) if True, stop
    '''
    self.data = data


  def _init_visualization(self):
    if not self.enabled: return

    self.fig  = plt.figure()
    self.axes = plt.gca()

    self.init_plot()

    self.fig.canvas.set_window_title(f'Visualization {self.name}')
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()


    if self.plot_only:
      plt.ion()
      plt.draw()
      plt.show()


  def stop_vis(self):
    print('call stop_vis')
    if not self.enabled: return
    self.process('exit') # send 
    self.stop()
    plt.close(self.fig) # close open matplotlib window
    
    self.shutdown = True
    self.enabled = False


