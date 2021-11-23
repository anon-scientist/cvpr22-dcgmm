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

from model                     import Model

from spn.algorithms.LearningWrappers            import learn_parametric, learn_classifier
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian, Bernoulli
from spn.structure.Base                         import Context


class Sum_Product_Network(Model):
  
  def build(self): return None, None
  
  def train(self, data, epochs):
    
    training_data = data.images[:1000]
    print('learn SPN classifier with training data shape:', training_data.shape)
    
    self.spn_classification = learn_classifier(
      data              = training_data,
      ds_context        = Context(parametric_types=[Gaussian] * training_data.shape[1]).add_domains(training_data),
      spn_learn_wrapper = learn_parametric, 
      label_idx         = 2,
      )
    print('Done')
