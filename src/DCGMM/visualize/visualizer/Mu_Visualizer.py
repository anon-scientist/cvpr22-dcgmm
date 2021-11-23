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
from matplotlib.widgets import CheckButtons
from visualize.visualizer.Visualizer import Visualizer

class Mu_Visualizer(Visualizer):

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
    self.ax_img            = self.axes.imshow(empty, cmap='gray', vmin=0, vmax=1.0)
    self.txt               = None
    self.txt_enum          = None
    self.txt_counter       = None

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

    # create checkbox to show/hide pis
    self.activated = False
    ax             = self.fig.add_axes([.77, .82, 0.3, .2], frame_on=False)
    self.chxbox    = CheckButtons(ax, ['heatmap'], [self.activated])
    def callback(_): self.activated = not self.activated
    self.chxbox.on_clicked(callback)


    # create checkbox to show/hide bmu counter
    self.activated_bmu = False
    ax_              = self.fig.add_axes([.77, .72, 0.3, .2], frame_on=False)
    self.chxbox_bmu = CheckButtons(ax_, ['bmu counter'], [self.activated_bmu])
    def callback_bmu(_): self.activated_bmu = not self.activated_bmu
    self.chxbox_bmu.on_clicked(callback_bmu)

    # create checkbox to show/hide indexing counter
    self.activated_index = False
    ax_                  = self.fig.add_axes([.77, .62, 0.3, .2], frame_on=False)
    self.chxbox_index    = CheckButtons(ax_, ['index'], [self.activated_index])
    def callback_index(_): self.activated_index = not self.activated_index
    self.chxbox_index.on_clicked(callback_index)


  def update_plot(self, data):
    if not self.enabled: return

    mus_pis         = data[0] # split {mus:..., pis:...} and [bmu, labels]
    bmu             = data[1] if len(data) >= 2 else None
    labels          = data[2] if len(data) >= 3 else None

    mus             = mus_pis[0].get('mus')
    pis             = mus_pis[0].get('pis')
    iteration       = mus_pis[0].get('iteration')

    if self.norm:
      min_   = np.min(mus)
      max_   = np.max(mus)
      images = (mus - min_) / (max_ - min_)
    else   : images = mus

    # combine all images to one
    image = images.reshape(*self.origin_shape)
    image = image.swapaxes(1, 2)
    image = image.reshape(*self.new_shape)
    image = self._add_grid(image)
    #image = 1 - image

    self.ax_img.set_array(image)
    self._enumerate()
    if labels is not None and len(labels) != 0: self._labels(labels)
    if bmu    is not None and len(bmu)    != 0: self._bmu_counter(bmu)
    self._heatmap(pis)

    try   : self.fig.canvas.flush_events()
    except:
      self.enabled = False
      self.stop()
    self.fig.canvas.draw_idle()

    if not self.plot_only:
      self.fig.savefig(f'{self.output_dir}{self.name}_{iteration}.pdf')

  def _bmu_counter(self, bmu):
    if bmu is None: return

    if not self.activated_bmu:
      if hasattr(self, 'bmu_') and self.txt_counter is not None:
        for txt in self.txt_counter: txt.remove()
        delattr(self, 'bmu_')
        delattr(self, 'txt_counter')
      return
    else:
      self.bmu_ = True
      bmu_counter = bmu[0].get('bmu_counter')
      bmu_counter = bmu_counter.reshape(self.y_plots, self.x_plots)
      bmu_counter = bmu_counter.T

      bmu_counter = bmu_counter - np.median(bmu_counter)
      bmu_counter = bmu_counter.astype(int)

      if hasattr(self, 'txt_counter') and self.txt_counter and len(self.txt_counter) > 0:
        for txt in self.txt_counter: txt.remove()
      self.txt_counter = list()

      it = np.nditer(bmu_counter, flags=['multi_index'])
      while not it.finished:
        x, y = it.multi_index
        x = x * self.w + (x + 1)
        y = y * self.h + (y + 1) + (self.h // 2 - 1)
        self.txt_counter += [self.axes.text(x, y, it[0], fontsize=10,  c='red' if it[0] < 0 else 'lime')]
        it.iternext()


  def _enumerate(self, enum=True):
    if not enum: return

    if not self.activated_index:
      if hasattr(self, 'index_') and self.txt_enum is not None:
        for txt in self.txt_enum: txt.remove()
        delattr(self, 'index_')
      return
    else:
      self.index_ = True

    if self.txt_enum:
      try:
        for txt_enum_ in self.txt_enum: txt_enum_.remove()
      except: pass
    self.txt_enum = list()

    it = np.nditer(np.arange(100).reshape(10, 10).T, flags=['multi_index'])
    while not it.finished:
      x, y = it.multi_index
      x = x * self.w + (x + 1)
      y = y * self.h + (y + 1) + (self.h // 2 - 1)
      self.txt_enum += [self.axes.text(x, y, it[0], fontsize=10,  c='yellow')]
      it.iternext()


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
      y = y * self.h + (y + 1) + (self.h // 2 - 1)
      self.txt += [self.axes.text(x, y, it[0], fontsize=16,  c='red')]
      it.iternext()


  def _heatmap(self, heat): # this only work, if pis are adapted and not the same for all components!
    if heat is None: return

    if not self.activated:
      if hasattr(self, 'heat'):
        self.heat.remove()
        delattr(self, 'heat')
      return

    min_, max_     = np.min(heat), np.max(heat)
    heat           = heat if min_ == max_ else (heat - min_) / (max_ - min_)
    heat           = heat.reshape(self.x_plots, self.y_plots)
    x_w, y_w       = (self.ax_img.get_array().shape)
    heat           = cv2.resize(heat, (x_w, y_w), interpolation=cv2.INTER_CUBIC) # cv2.INTER_NEAREST

    if not hasattr(self, 'heat'):
      self.heat = self.axes.imshow(
        heat                     ,
        interpolation = 'bicubic',
        cmap          = 'jet'    ,
        alpha         = .5       ,
        )

    if not hasattr(self, 'cbar'):
      self.cbar = self.fig.colorbar(
        self.heat              ,
        ax        = self.axes ,
        drawedges = False      ,
        format    = '%.3f'     ,
        ticks     = [0, 0.5, 1],
        label     = 'pis'      ,
        )

    self.heat.set_data(heat) # update the heat map
    self.cbar.draw_all()
    self.cbar.set_alpha(1)   # avoid lines caused by transparency
    cbar_ticks = [ float(f'{x:.4f}') for x in np.linspace(min_, max_, num=3, endpoint=True ) ]
    self.cbar.ax.set_yticklabels(cbar_ticks)



if __name__ == '__main__':
  while True:
    try:
      vis = Mu_Visualizer(
        #server_address = '192.168.41.10',
        #server_port    = 443           ,
        server_port    = 10000           ,
        name           = 'MUS'         ,
        from_layers    = [
          ['L2_gmm'   , 'mus,pis'     ],
          #['L5_linear', 'proto_labels'],
          ['L2_gmm'   , 'bmu'         ],
          ['L3_linear', 'proto_labels'],
          ]                            ,
        )
      vis.go() # blocking
    except Exception as ex:
      raise ex
      try   : vis.stop_vis()
      except: pass
      print('restart Vis')


