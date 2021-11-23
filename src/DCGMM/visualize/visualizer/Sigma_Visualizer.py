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
from matplotlib.widgets                 import CheckButtons
from visualize.visualizer.Visualizer    import Visualizer

class Sigma_Visualizer(Visualizer):

  def __init__(self, **kwargs):
    ''' construct an preconfigured visualization object  '''
    self.from_layers = kwargs.get('from_layers')
    self.norm        = kwargs.get('norm', True)
    self.data        = None
    super().__init__(**kwargs)

  def go(self):
    while True:
      self.result_available.wait()
      self.update_plot(self.data)
      self.result_available.clear()


  def select_config(self):
    gmm_conf       = self.from_layers[0] # conf of linear layer not used
    gmm_layer_name = gmm_conf[0]
    gmm_conf       = self.received_config.get(gmm_layer_name)
    self.w         = gmm_conf.get('width'   )
    self.h         = gmm_conf.get('height'  )
    self.c         = gmm_conf.get('channels')
    self.convMode  = gmm_conf.get('convMode')
    self.x_plots   = gmm_conf.get('n'       )
    self.y_plots   = gmm_conf.get('n'       )
    self.D         = gmm_conf.get('c_in'    )
    self.h_out     = gmm_conf.get('h_out'   )
    self.w_out     = gmm_conf.get('w_out'   )
    if self.convMode:
      self.h_out = self.w_out = 1


  def init_plot(self):
    self.select_config()

    self.origin_shape      = (self.y_plots, self.x_plots, self.h, self.w)   if self.c == 1 else (self.y_plots, self.x_plots, self.h, self.w, self.c)
    self.new_shape         = (self.y_plots * self.h, self.x_plots * self.w) if self.c == 1 else (self.y_plots * self.h, self.x_plots * self.w, self.c)
    empty                  = np.zeros(self.new_shape)
    if self.c == 1: empty[0, 0]    = 1. # grayscale: it feels like shit
    else          : empty[0, 0, :] = 1. # color
    empty                  = self._add_grid(empty)
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

    self.activated = False

    # create checkbox to show/hide pis
    ax             = self.fig.add_axes([.77, .82, 0.3, .2], frame_on=False)
    self.chxbox    = CheckButtons(ax, ['heatmap'], [self.activated])
    def callback(_): self.activated = not self.activated
    self.chxbox.on_clicked(callback)


  def update_plot(self, data):
    if not self.enabled: return
    data_           = data[0] # split {mus:..., pis:...} and [labels]
    labels          = data[1] if len(data) == 2 else None
    sigmas          = data_[0].get('sigmas')
    iteration       = data_[0].get('iteration')

    min_   = np.min(sigmas)
    max_   = np.max(sigmas)
    print('min_', min_, 'max_', max_)
    images = 1 / sigmas
    if self.norm: images = (sigmas - min_) / (max_ - min_)
    else        : images = sigmas



    # combine all images to one
    image = images.reshape(*self.origin_shape)
    image = image.swapaxes(1, 2)
    image = image.reshape(*self.new_shape)
    image = self._add_grid(image)
    image = 1 - image

    self.ax_img.set_array(image)
    if labels is not None and len(labels) != 0: self._labels(labels)

    try   : self.fig.canvas.flush_events()
    except:
      self.enabled = False
      self.stop()
    self.fig.canvas.draw_idle()

    if not self.plot_only:
      self.fig.savefig(f'{self.output_dir}{self.name}_{iteration}.pdf')


  def _labels(self, labels):
    if labels is None: return
    labels = labels.reshape(self.y_plots, self.x_plots)
    labels = labels.T
    if self.txt:
      for txt in self.txt: txt.remove()
    self.txt = list()

    it = np.nditer(labels, flags=['multi_index'])
    while not it.finished:
      x, y = it.multi_index
      x = x * self.w + (x + 1)
      y = y * self.h + (y + 1) + (self.h // 2 + 1)
      self.txt += [self.axes.text(x, y, it[0], fontsize=15,  c='red')]
      it.iternext()


if __name__ == '__main__':
  while True:
    try:
      vis = Sigma_Visualizer(
        #server_address = '192.168.41.10',
        server_port    = 10000           ,
        name           = 'SIGMAS'      ,
        from_layers    = [
          ['L2_gmm'   , 'sigmas'      ],
          ['L3_linear', 'proto_labels'],
          ]                            ,
        )
      vis.go() # blocking
    except:
      try   : vis.stop_vis()
      except: pass
      print('restart Vis')


