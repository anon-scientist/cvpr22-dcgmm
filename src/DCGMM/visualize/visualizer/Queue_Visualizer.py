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
import numpy as np
from .Visualizer import Visualizer
from .Queue import Queue
import warnings

class Queue_Visualizer(Visualizer):

  def __init__(self, name, **kwargs):
    ''' construct an preconfigured visualization object  '''
    print('init')
    super(Queue_Visualizer, self).__init__(name=name, **kwargs)
    print('init done')
    self.figsize           = kwargs.get('figsize'   , (16, 4)               )
    self.xlim              = kwargs.get('xlim'      , (None, None)          )
    self.ylim              = kwargs.get('ylim'      , (None, None)          )
    self.xlabel            = kwargs.get('xlabel'    , ''                    )
    self.ylabel            = kwargs.get('ylabel'    , ''                    )
    self.queue_size        = kwargs.get('queue_size', 1000                  )
    self.queue             = kwargs.get('queue'     , Queue(self.queue_size))
    self.linestyle         = kwargs.get('linestyle', '-'                    )
    self.color             = kwargs.get('color'    , 'black'                )
    self.label             = kwargs.get('label'    , self.name              )


  def init_plot(self):
    print('init plot')
    if not self.enabled: return

    if self.queue and self.xlim == (None, None): self.xlim = (0, self.queue.max_values) # auto resize mechanism for queues

    self.axes.set_xlim(
      self.xlim[0] if self.xlim[0] is not None else 0, # set x-axis limits (left)
      self.xlim[1] if self.xlim[1] is not None else 1, # set x-axis limits (right)
      )
    self.axes.set_ylim(
      self.ylim[0] if self.ylim[0] is not None else 0, # set y-axis limits (lower)
      self.ylim[1] if self.ylim[1] is not None else 1, # set y-axis limits (upper)
      )

    self.axes.set_xlabel(self.xlabel)
    self.axes.set_ylabel(self.ylabel)

    if self.queue:
      self.label = f'{self.label} (smoothing factor={self.queue.alpha})'

    self.line, = self.axes.plot( # placeholder (is updated in redraw)
      [0]                   ,  # xs
      [0]                   ,  # ys
      linestyle = self.linestyle ,
      color     = self.color     ,
      label     = self.label     ,
      )



  def update_plot(self, data):
    if not self.enabled: return

    iteration = data.get('iteration')
    data      = data.get('data')
    self.queue.add(data)

    if self.queue: # use the data of the queue, if available
      ys = self.queue._data()[::-1]
      xs = np.arange(len(self.queue._data()[::-1]))

    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      self.axes.set_xlim(
        self.axes.get_xlim()[0] if self.xlim[0] is not None else min(xs), # define x-axis limits (left limit)
        self.axes.get_xlim()[1] if self.xlim[1] is not None else max(xs), # define x-axis limits (right limit)
        )
      self.axes.set_ylim(
        self.axes.get_ylim()[0] if self.ylim[0] is not None else min(ys), # define y-axis limits (lower limit)
        self.axes.get_ylim()[1] if self.ylim[1] is not None else max(ys), # define y-axis limits (upper limit)
        )

    self.line.set_xdata(xs)
    self.line.set_ydata(ys)

    try   : self.fig.canvas.flush_events()
    except: self.enabled = False

    if not self.plot_only:
      self.fig.savefig(f'{self.output_dir}{self.name}_{iteration}.pdf')



