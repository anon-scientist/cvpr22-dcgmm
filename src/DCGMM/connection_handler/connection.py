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
''' client connection '''
import threading
import queue
from DCGMM.utils import log
import struct
import json

MAX_CONNECTION_ERRORS = 3

class Connection(threading.Thread):

  def __init__(self, connection, client_address, **kwargs):
    ''' class (thread) that handles/represents a client connection
    @param connection: socket for sending/receiving data on the client connection
    @param client_address: client socket address
    '''
    threading.Thread.__init__(self, daemon=True)
    self.socket                = connection
    self.client_address        = client_address
    self.setName(client_address)
    self.condition             = threading.Condition(threading.Lock())
    self.data_queue            = queue.Queue()  # client-specific data queue

    self.num_connection_errors = 0

    self.layer_params          = kwargs.get('layer_params'      )
    self.environment_params    = kwargs.get('environment_params')


  def _enqueue(self, data):
    ''' enqueue data
    @param data: data to put in the queue
    '''
    self.data_queue.put(data)


  def receive_config(self):
    ''' receive the configuration from the visualizer client '''
    config_dict = self.socket.recv(1024).decode()
    self.config = json.loads(config_dict)
    log.debug(f'received configuration {self.config}')


  def fetch_config(self):
    ''' fetch configurations from the corresponding layers '''
    layers              = [ layer[0] for layer in self.config.get('from_layers') ]                                                       # define layers
    self.config_to_send = { layer_name: layer_params for layer_name, layer_params in self.layer_params.items() if layer_name in layers } # select configs from layers
    if 'environment' in layers: # e.g. for sampling
      environment_parameter = { k: self.environment_params.get(k) for k in [
        'sampling_batch_size',
        'nr_sampling_batches',
        'w', 'h', 'c'        ,
        ]}
      self.config_to_send['environment'] = environment_parameter

    log.debug(f'config for {self} visualization {self.config_to_send}')


  def send_config(self):
    ''' send the layer configurations back to the client '''
    self.socket.send(json.dumps(self.config_to_send).encode())
    log.debug(f'send config for {self} visualization {self.config_to_send}')


  def send_data(self):
    ''' dequeue data, determine its length and send both to the client '''
    data = self.data_queue.get()
    log.debug(f'send_data_bytes: {len(data)} bytes to {self.client_address}')
    try:
      num_bytes = len(data)
      num_bytes = struct.pack('!I', num_bytes)
      all_      = num_bytes + data
      self.socket.send(all_)
    except Exception as ex:
      log.warning('Exception: {}'.format(str(ex).replace('\n',''))) # contains newlines...bug in python3.7.x
      self.num_connection_errors += 1
      del(ex)


  def run(self):
    ''' loop for sending available preprocessed data to the connected client '''
    self.receive_config() # receive initial config
    self.fetch_config()
    self.send_config()

    while True:
      if self.num_connection_errors >= MAX_CONNECTION_ERRORS: # maximum connection errors exceeded -> terminate
        log.error(f'connection retries exceeded: exit ({self.client_address[0]}:{self.client_address[1]})')
        break

      if not self.data_queue.empty(): # data available -> send it
        self.send_data()
        continue

      log.debug(f'no data in queue...goto sleep ({self.client_address})')
      try: # wait until data is available
        self.condition.acquire()
        self.condition.wait()
      finally:
        self.condition.release()

    try:
      self.socket.close()
    except Exception as ex:
      log.warning(f'could not close connection: {ex}')
    del(self.socket)
