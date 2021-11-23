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
''' load mus (numpy files) and plot '''

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

file = 'mus'

data = np.load(f'{file}.npy').squeeze()
data = interp1d([np.min(data), np.max(data)], [0., 1.])(data)
print('data.shape', data.shape)
print('min, max', data.min(), data.max())


w = 28
h = 28

if 'patch' in file: w = h = 6

c = 1

x = 8
y = 8


f, axes = plt.subplots(y, x)
axes = axes.ravel()

for (d, ax_) in zip(data, axes):
  ax_.imshow(d.reshape(w, h, c) if c == 3 else d.reshape(w, h), cmap='gray') # reversed colormap (e.g., gray_r)
  ax_.tick_params( # disable labels and ticks
    axis        = 'both',
    which       = 'both',
    bottom      = False ,
    top         = False ,
    left        = False ,
    right       = False ,
    labelbottom = False ,
    labelleft   = False ,
    )

plt.tight_layout(pad=0, h_pad=.4, w_pad=-10)
plt.savefig(f'mus_{file}.pdf')
