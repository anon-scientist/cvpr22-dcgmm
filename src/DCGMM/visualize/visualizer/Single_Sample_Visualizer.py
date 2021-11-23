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

class Single_Sample_Visualizer(Visualizer):

  def __init__(self, name, **kwargs):
    ''' construct an preconfigured visualization object  '''
    super(Single_Sample_Visualizer, self).__init__(name, **kwargs)

  def init_plot(self):
    self.origin_shape      = (self.y_plots, self.x_plots, self.h, self.w)   if self.c == 1 else (self.y_plots, self.x_plots, self.h, self.w, self.c)
    self.new_shape         = (self.y_plots * self.h, self.x_plots * self.w) if self.c == 1 else (self.y_plots * self.h, self.x_plots * self.w, self.c)
    empty                  = np.zeros(self.new_shape)

    if self.c == 1: empty[0, 0]    = 1. # grayscale
    else          : empty[0, 0, :] = 1. # color
    empty                  = self.add_grid(empty)
    self.ax_img            = self.axes.imshow(empty, cmap='gray')
    self.txt               = None

    self.axes.tick_params( # disable labels and ticks
      axis        = 'both',
      which       = 'both',
      bottom      = False ,
      top         = False ,
      left        = False ,
      right       = False ,
      labelbottom = False ,
      labelleft   = False ,
      )


  def visualize_sample(self, sample, label=None):
    ''' visualize a single sample '''
    if not self.enabled: return
    image = sample.squeeze()
    image = image.reshape((self.h, self.w))
    self.ax_img.set_array(image)
    if label:
      if self.txt: self.txt.remove()
      self.txt = self.axes.text(-2, -2, label, fontsize=30,  c='red')
    self.fig.canvas.flush_events()
    self.fig.canvas.draw_idle()

