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
''' handler for client connections '''

import threading
import socket
import pickle
import zlib
from . import Connection
from DCGMM.utils.log          import log
from collections        import defaultdict

class Connection_Handler(threading.Thread):

  def __init__(self, **kwargs):
    ''' class (thread) that manages client connections '''
    self.client_threads     = list()

    self.HOST               = kwargs.get('interface', '0.0.0.0')  # listen on all interfaces
    self.PORT               = kwargs.get('port'     , 10000      )  # port for new connections

    self.environment_params = { k:v for k, v in kwargs.get('environment_params').items() } # make a copy
    self.layer_params       = {}
    model                   = kwargs.get('model')

    from model.Aggr_Stacked_GMM import Aggr_Stacked_GMM
    if model and isinstance(model, Aggr_Stacked_GMM):
      ''' Update model vars for each layer, since we have up to N models with M layers '''
      for key, gmm in model.lookup.items():
        layers = gmm.layers
        self.layer_params = { key: { layer.get_name(): layer.fetch_variables_conf() for layer in layers } }
        self.environment_params.update(vars(kwargs.get('model')))

    elif model:
      layers                  = model.layers
      self.layer_params       = { layer.get_name(): layer.fetch_variables_conf() for layer in layers }
      self.environment_params.update(vars(kwargs.get('model')))                              # add model parameter

    threading.Thread.__init__(self, daemon=True)

    self.socket = socket.socket()
    try   : self.socket.bind((self.HOST, self.PORT))
    except Exception as ex:
      log.warning(f'could not bind connection handler on {self.HOST}:{self.PORT}')
      raise ex

    self.socket.listen()


  def fetch_and_send(self, layer_list, *data_list, **kwargs):
    ''' send data to active client threads TODO: rework this ugly code!

    @param environment: reference to the calling class
    @param layer_list : list of all layers from where to fetch the data
    @param *data_list : list of addition data points tuple(<name>, data), e.g., generated samples
    @param **kwargs   : parameter from calling method for the method visualize of the layers
    '''
    # 1. collect all needed data from the active client threads
    data_select_dict = defaultdict(list)
    for client_thread in self.client_threads:
      for layer_name, data_name in client_thread.config.get('from_layers'): data_select_dict[layer_name] += [data_name] # collect all selectors for all layers
    for layer_name, data_names in data_select_dict.items(): data_select_dict[layer_name] = list(set(data_names))        # make them unique
    # 2. fetch all needed data from the lists
    data_dict        = dict()
    for layer in layer_list:
      if layer.get_name() in data_select_dict.keys():
        data_ = layer.share_variables(**kwargs, **{ k:True for k in data_select_dict.get(layer.get_name()) }) # collect all needed data from the layers
        if data_ is not None: data_dict[layer.get_name()] = data_
    # 3. add additional data from the environment
    data_dict['environment'] = dict()
    for var_name, data_ in data_list:
      data_dict['environment'][var_name] = data_                            # TODO: fix unchanged update (changed reference? trigger? ...?)
    if len(data_dict['environment']) == 0: del data_dict['environment']     # if contains no data
    # 4. send data to individual network clients
    for client_thread in self.client_threads:
      data = list()
      for from_layer, key_ in client_thread.config.get('from_layers'):
        layer_data_dict = data_dict.get(from_layer)
        if layer_data_dict is not None: data += [layer_data_dict.get(key_)] # check if data from layer is available (because "environment" can be empty but requested by visualizer)
      if len(data) == 0: continue                                           # if no data available, skip (because "environment" can be empty but requested by visualizer)
      data = zlib.compress(pickle.dumps(data))
      client_thread._enqueue(data)
    self.wakeup()


  def wakeup(self):
    ''' wake up all client threads, remove dead client connections/threads '''
    log.debug(f'wake up all client threads ({len(self.client_threads)})')
    for client_thread in self.client_threads:
      if not client_thread.is_alive():
        log.warning(f'remove thread from thread ({client_thread}) list')
        self.client_threads.remove(client_thread)
        continue
      try:
        client_thread.condition.acquire()
        client_thread.condition.notify()
      finally:
        client_thread.condition.release()


  def has_clients(self):
    return len(self.client_threads) > 0


  def run(self):
    ''' loop for handling incoming client connections '''
    log.info(f'start Connection Handler for {self.HOST}:{self.PORT}')
    while True:
      try:
        connection, client_address = self.socket.accept()
        log.info(f'got new connection from {client_address[0]}:{client_address[1]}')
        client_thread = Connection(connection, client_address, layer_params=self.layer_params, environment_params=self.environment_params)
        client_thread.start()
        self.client_threads += [client_thread]
      except Exception as ex:
        log.warning(f'could not create connection: {ex}')
        break


    self.socket.close()
