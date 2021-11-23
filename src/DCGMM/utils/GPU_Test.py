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
'''
Disable TensorFlow error messages # Test if a GPU is available (disabled)
'''

import os
import sys
import logging
from absl import logging as absl_logging
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

fmt = '[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s'
formatter = logging.Formatter(fmt)

absl_logging.get_absl_handler().setFormatter(formatter)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tf debug messages

import tensorflow as tf
from tensorflow.python.client import device_lib
#from . import gpu

for h in tf.get_logger().handlers:
  h.setFormatter(formatter)

tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')
if not gpus: print('No valid gpu found', file=sys.stderr)
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


