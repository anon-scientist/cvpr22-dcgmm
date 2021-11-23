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

## central DCGMM layer classes. Batch_Normalization is deprecated, not sure 
## whether it even works. Not needed since GMM layer output responsibilities
## which are normalized and bounded.

## only library classes, no "executables" here

from .Layer                     import Layer
from .GMM_Layer                 import GMM_Layer
from .Input_Layer               import Input_Layer
from .Folding_Layer             import Folding_Layer
from .MaxPooling_Layer          import MaxPooling_Layer
from .Linear_Classifier_Layer   import Linear_Classifier_Layer
from .regularizer               import Regularizer
