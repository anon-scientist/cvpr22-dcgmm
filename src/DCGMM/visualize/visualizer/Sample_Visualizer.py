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
import math
import numpy as np
from visualize.visualizer import Visualizer

class Sample_Visualizer(Visualizer):

  def __init__(self, **kwargs):
    ''' construct an preconfigured visualization object  '''
    self.from_layers = kwargs.get('from_layers')
    self.data        = None
    super().__init__(**kwargs)

    while True:
      self.result_available.wait()
      self.update_plot(self.data)
      self.result_available.clear()

  def select_config(self):
    environment                 = self.from_layers[0]
    environment                 = environment[0]
    environment                 = self.received_config.get(environment)
    self.w                      = environment.get('w')
    self.h                      = environment.get('h')
    self.c                      = environment.get('c')
    self.sampling_batch_size    = environment.get('sampling_batch_size' , 100)
    self.nr_sampling_batches    = environment.get('nr_sampling_batches') if environment.get('nr_sampling_batches') else 1 
    
    self.num_samples            = self.sampling_batch_size * self.nr_sampling_batches
    self.x_plots = self.y_plots = int(math.sqrt(self.num_samples))
    self.D                      = self.w * self.h * self.c
    self.h_out                  = 1
    self.w_out                  = 1


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


  def _labels(self, labels):
    if labels is None: return
    labels = labels.reshape(self.y_plots, self.x_plots)
    labels = labels.T
    if self.txt:
      for txt in self.txt: txt.remove()
    self.txt = list()

    it = np.nditer(labels, flags=['multi_index'])
    while not it.finished:
      x, y      = it.multi_index
      x         = x * 28 + (x + 1)
      y         = y * 28 + (y + 1) + 13
      self.txt += [self.axes.text(x, y, it[0], fontsize=15,  c='red')]
      it.iternext()


  def update_plot(self, data):
    if not self.enabled: return
    samples, labels = data
    norm            = False
    if norm:
      min_   = np.min(samples)
      max_   = np.max(samples)
      images = samples if min_ == max_ else (samples - min_) / (max_ - min_)
    else   : images = samples

    # combine all images to one
    image = images.reshape(*self.origin_shape)
    image = image.swapaxes(1, 2)
    image = image.reshape(*self.new_shape)
    image = self._add_grid(image)
    #image = 1 - image

    self.ax_img.set_array(image)
    self._labels(labels)

    try   : self.fig.canvas.flush_events()
    except Exception as ex:
      raise ex
      self.enabled = False
      self.stop()
    self.fig.canvas.draw_idle()

    if not self.plot_only:
      self.fig.savefig(f'{self.output_dir}{self.name}_.pdf') # TODO: add iteration


if __name__ == '__main__':

  Sample_Visualizer(
    name           = 'Generated'         ,
    server_port    = 10000                 ,
    from_layers    = [
      ['environment', 'generate'       ] ,
      ['environment', 'generate_labels'] ,
      ]                                  ,
    )

