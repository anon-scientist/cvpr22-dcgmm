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

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D # no not remove, used for 3d plots
import numpy as np
from scipy.interpolate import interp1d
from .Visualizer import Visualizer

class Pi_Visualizer(Visualizer):

  def __init__(self, name, **kwargs):
    ''' construct an preconfigured visualization object  '''
    super(Pi_Visualizer, self).__init__(name, **kwargs)

  def init_plot(self):
    self.origin_shape      = (self.y_plots, self.x_plots, self.h, self.w)   if self.c == 1 else (self.y_plots, self.x_plots, self.h, self.w, self.c)
    self.new_shape         = (self.y_plots * self.h, self.x_plots * self.w) if self.c == 1 else (self.y_plots * self.h, self.x_plots * self.w, self.c)
    empty                  = np.zeros(self.new_shape)
    if self.c == 1: empty[0, 0]    = 1. # grayscale: it feels like shit
    else          : empty[0, 0, :] = 1. # color
    self.ax_img            = self.axes.imshow(empty, cmap='gray')

    # 3d plot variables
    self.bar               = None
    self.cbar              = None
    self.ax                = None
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


  def visualize_pis(self, pis, **kwargs):
    if not self.enabled: return

    it         = kwargs.get('iteration')
    min_, max_ = np.min(pis), np.max(pis)

    if self.store_only_output:
      self.values[it] = pis
      return
    if not self.plot_only: return

    np.save(f'./plots/pis_{it}.npy', pis) # store numpy files for recreation (e.g., rotate image)

    if self.bar: # clear figure after each update
      self.bar.remove()
      self.ax.remove()
      self.cbar.remove()
    else:
      self.fig.clf()
    self.ax = self.fig.add_subplot(111, projection='3d')

    # create bar values (sidewalls)
    _x        = np.arange(self.w)
    _y        = np.arange(self.h)
    _xx, _yy  = np.meshgrid(_x, _y)
    x  , y    = _xx.ravel(), _yy.ravel()
    depth     = 1
    width     = 1

    # create bar values (bottom and top)
    z         = interp1d([min_, max_], [0.0, 1.])(pis)
    bottom    = np.zeros_like(z)

    # define colors
    newcmp    = matplotlib.cm.get_cmap('Blues', 256)
    newcmp    = ListedColormap(newcmp(np.linspace(0.4, .9, 256))) # slice 'Blues' color map
    norm      = matplotlib.colors.Normalize(min_, max_)           # normalize color function
    colors    = newcmp(norm(z))                                   # apply color normalizer

    # create 3d plot
    self.bar  = self.ax.bar3d(x, y, bottom, width, depth, z, color=colors, shade=True, antialiased=True)

    self.ax.set_title('pis')

    # plot view orientation
    self.ax.view_init(25, 135)
    plt.subplots_adjust(top=0.5) # vertical compressing

    # define z axis limits
    self.ax.set_zlim3d(0, 1)
    z_scale   = np.linspace(0, 1, 5) # 5 ticks
    self.ax.set_zticks(z_scale)

    # disable x and y axis labels
    self.ax.set_yticklabels([])
    self.ax.set_xticklabels([])

    # define z axis lables
    z_label_vals = np.linspace(min_, max_, 5)
    z_label      = [ f'{float(val):.3}' for val in z_label_vals ] # only three decimal digits
    self.ax.set_zticklabels(z_label)

    # create colorbar
    m         = matplotlib.cm.ScalarMappable(cmap=newcmp)
    self.cbar = plt.colorbar(m, shrink=.7, ticks=z_scale, pad=0.1)
    self.cbar.ax.set_yticklabels(z_label)

    for rotation in range(0, 360, 45):
      self.ax.view_init(25, rotation)
      self.fig.savefig(f'./plots/pis_{it}_rot_{rotation}.pdf', format='pdf', bbox_inches='tight')
      self.fig.savefig(f'./plots/pis_{it}_rot_{rotation}.png', format='png', bbox_inches='tight')

