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
from .Stacked_GMM import Stacked_GMM
import tensorflow as tf
import numpy as np


class Stacked_GMM_Outliers(Stacked_GMM):

  def __init__(self, init_from_kwargs=True, **kwargs):
    super(Stacked_GMM_Outliers, self).__init__(init_from_kwargs=True, **kwargs)
    parser = self.parser 
    self.outlier_detection_layer = parser.add_argument('--outlier_detection_layer', type=int, default=-1, required=False, help='layer index of layer that is used for outlier dteection')


  def get_outlier_loss(self,xs):
    self.forward(xs=xs) ;
    lss = self.layers[self.outlier_detection_layer].loss(self.forwards[self.outlier_detection_layer],ys=None)
    return tf.reduce_mean(lss) 

  def do_inpainting(self, xs, **kwargs):
    pass


  def test_step(self, xs, ys, **kwargs):
    tmp = Stacked_GMM.test_step(self, xs,ys,**kwargs)
    lss = tf.reduce_mean(self.raw_losses[self.outlier_detection_layer], axis=(1,2)) ;
    tmp['outliers']={"scores":lss.numpy()}
    
    return tmp ;


