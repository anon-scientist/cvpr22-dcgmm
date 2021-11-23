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
from visualize.visualizer.Visualizer import Visualizer
from visualize.visualizer.Queue      import Queue
import numpy as np
import warnings

class Network_Queue_Visualizer(Visualizer):

  colors     = [ 'r', 'g', 'b', 'k' ]
  linestyles = [ '-', '-', '-', '-' ]

  def __init__(self, **kwargs):
    ''' construct an preconfigured visualization object  '''
    self.from_layers = kwargs.get('from_layers')
    self.norm        = kwargs.get('norm', False)
    
    self.figsize     = kwargs.get('figsize'   , (16, 4)               )
    self.xlim        = kwargs.get('xlim'      , (None, None)          )
    self.ylim        = kwargs.get('ylim'      , (None, None)          )
    self.xlabel      = kwargs.get('xlabel'    , ''                    )
    self.ylabel      = kwargs.get('ylabel'    , ''                    )
    self.queue_size  = kwargs.get('queue_size', 3000                  )
    
    self.queues      = [ Queue(self.queue_size) for _ in self.from_layers ]
    
    self.data        = None
    super().__init__(**kwargs)
  
  
  def go(self):
    while True:
      self.result_available.wait()
      self.update_plot(self.data)
      self.result_available.clear()


  def select_config(self):
    pass


  def init_plot(self):
    self.select_config()
    
    self.axes.set_xlabel(self.xlabel)
    self.axes.set_ylabel(self.ylabel)
    
    self.axes.set_xlim(0, self.queue_size)

    self.lines = [    
      self.axes.plot( # placeholder (is updated in redraw)
        [0]                           ,  # xs
        [0]                           ,  # ys
        label     = label             ,   
        linestyle = self.linestyles[i],
        color     = self.colors[i]    ,
        )[0]
      for i, (_, label) in enumerate(self.from_layers) ]
    self.activated = False
    
    self.axes.legend()


  def update_plot(self, data):
    if not self.enabled: return
     
    for i, queue in enumerate(self.queues): 
      queue.add(data[i])
    self.axes.set_ylim(None, None)
    
    min_y, min_x, max_y, max_x = [], [], [], []
    for i, _ in enumerate(self.from_layers):
      ys = self.queues[i]._data()[::-1]
      xs = np.arange(len(self.queues[i]._data()[::-1]))
      
      min_y += [min(ys)]
      max_y += [max(ys)]
      min_x += [min(xs)]
      max_x += [max(xs)]
      
      self.lines[i].set_xdata(xs)
      self.lines[i].set_ydata(ys)
      
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      self.axes.set_xlim(
        self.axes.get_xlim()[0] if self.xlim[0] is not None else min(min_x), # define x-axis limits (left limit)
        self.axes.get_xlim()[1] if self.xlim[1] is not None else max(max_x), # define x-axis limits (right limit)
        )
      self.axes.set_ylim(
        self.axes.get_ylim()[0] if self.ylim[0] is not None else min(min_y), # define y-axis limits (lower limit)
        self.axes.get_ylim()[1] if self.ylim[1] is not None else max(max_y), # define y-axis limits (upper limit)
        )
    
    try   : self.fig.canvas.flush_events()
    except:
      self.enabled = False
      self.stop()
    self.fig.canvas.draw_idle()


if __name__ == '__main__':
  vis = Network_Queue_Visualizer(
    #server_address = '192.168.41.10',
    server_port    = 443           ,
    name           = 'Regularizer' ,
    #from_layers=[['environment', 'reg_long'], ['environment', 'reg_short'], ['environment', 'reg_diff'], ['environment', 'reg_time']],
    from_layers=[['environment', 'avgLong'] , ['environment', 'lastAvg']  , ['environment', 'l0']      , ['environment', 'limit']],
    )
  vis.go() # blocking
    

