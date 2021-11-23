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

import cv2
import numpy as np
from scipy.interpolate import interp1d
from .Visualizer import Visualizer

class Conv_Visualizer(Visualizer):

  def __init__(self, name, **kwargs):
    ''' construct an preconfigured visualization object  '''
    super(Conv_Visualizer, self).__init__(name, **kwargs)


  def init_plot(self):
    self.origin_shape      = (self.y_plots, self.x_plots, self.h, self.w)   if self.c == 1 else (self.y_plots, self.x_plots, self.h, self.w, self.c)
    self.new_shape         = (self.y_plots * self.h_out, self.x_plots * self.w_out) if self.c == 1 else (self.y_plots * self.h_out, self.x_plots * self.w_out, self.c)
    empty                  = np.zeros(self.new_shape)
    if self.c == 1: empty[0, 0]    = 1. # grayscale
    else          : empty[0, 0, :] = 1. # color
    empty                  = self._add_grid(empty)
    self.ax_img            = self.axes.imshow(empty, cmap='gray_r')

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


  def _add_grid(self, data, val=1.0):
    if self.grid: # add grid lines
      for i in range(self.x_plots + 1):
        data = np.insert(data, i * self.w_out  + i, val, axis=1)
        data = np.insert(data, i * self.h_out + i, val, axis=0)
    return data


  def update_plot(self, data):
    if not self.enabled: return
    iteration  = data.get('iteration')
    conv_masks = data.get('convMask')
    norm       = True

    if norm: heat = interp1d([np.min(conv_masks), np.max(conv_masks)], [0., 1.])(conv_masks)
    else   : heat = conv_masks
    heat           = heat.reshape([self.h, self.h, self.w, self.w])
    heat           = heat.swapaxes(1, 2)
    heat           = heat.reshape([self.D, self.D])

    heat           = cv2.resize(heat, (self.h_out * self.x_plots, self.w_out * self.y_plots), interpolation=cv2.INTER_CUBIC) # cv2.INTER_NEAREST
    heat           = interp1d([np.min(heat), np.max(heat)], [0., 1.])(heat)
    heat           = self._add_grid(heat, val=1.)
    self.ax_img.set_data(heat)

    try   : self.fig.canvas.flush_events()
    except: self.enabled = False
    self.fig.canvas.draw_idle()
    if not self.plot_only:
      self.fig.savefig(f'{self.output_dir}{self.name}_{iteration}.pdf')

