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
import logging

log           = logging.getLogger('GMM_Experiment')  # create logger
log.propagate = False # disable clash with tensorflow logger
#fh            = logging.FileHandler('gmm.log')       # create file handler which logs even debug messages
ch            = logging.StreamHandler()              # create console handler with a higher log level
formatter     = logging.Formatter('%(asctime)s, %(levelname)-7s [%(filename)-30s:%(lineno)-5d]: %(message)s', '%H:%M:%S') # create formatter and add it to the handlers
#fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
#log.addHandler(fh)
log.addHandler(ch)


def change_loglevel(log_level):
  ''' change the current log level

  @param log_level: log level (int) (see logging module)
  '''
  log.debug(f'change log level to: {log_level}')
  log.setLevel(log_level)
  #fh.setLevel(log_level)
  ch.setLevel(log_level)
