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
''' base class for streaming clients '''

import pickle
import zlib
from abc import ABC, abstractmethod
from socket import AF_INET, SOCK_STREAM, socket
import threading
import itertools
import struct
import time
from utils import log


class Stream_Client(ABC, threading.Thread):

  def __init__(self, **kwargs):
    threading.Thread.__init__(self)
    self.socket              = None
    self.running             = True
    self.result_available    = threading.Event()
    self.num_socket_timeouts = 0

    self.SOCKET_TIMEOUT      = 60 * 10  # seconds
    self.MAX_SOCKET_TIMEOUTS = 3
    self.BUFFER_SIZE         = 4096

    self.SERVER_ADDRESS      = kwargs.get('server_address', '127.0.0.1')
    self.SERVER_PORT         = kwargs.get('server_port'   , 11338      )


  @abstractmethod
  def process(self, data):
    ''' abstract process method is called by the thread itself (in run) (method stub)

    @param data: input data from stream server
    @return: if true, thread is stopped (bool), else: stop processing and disconnect from server (only used by file client)
    '''
    pass

  @abstractmethod
  def send_and_receive_config(self): pass


  def connect(self):
    ''' use constants to create a connect to the streaming server '''
    log.info(f'connect to {self.SERVER_ADDRESS}:{self.SERVER_PORT}')
    while True:
      try:
        client_socket = socket(AF_INET, SOCK_STREAM)
        client_socket.settimeout(self.SOCKET_TIMEOUT)
        client_socket.connect((self.SERVER_ADDRESS, self.SERVER_PORT))
        self.socket   = client_socket
        self.send_and_receive_config()
        break
      except:
        time.sleep(1)


  def stop(self):
    ''' stop the processing loop '''
    self.running = False


  def _disconnect(self):
    ''' close the connection to the streaming server '''
    log.info('exit')
    self.socket.close()


  def run(self):
    ''' loop for receiving data and start the processing step '''

    while self.running:
      try:
        log.debug('wait for data...')
        data = self.socket.recv(1500)
        num_bytes  = struct.unpack('!I', data[:4])[0]
        data       = data[4:]
        all_data   = [data]
        num_bytes -= len(data)
        while num_bytes > 0:  # load data until a full data block was received
          data = self.socket.recv(1500 if num_bytes > 1500 else num_bytes)
          all_data  += [data]
          num_bytes -= len(data)
        log.debug('unpack data...')
        all_data_flat = list(itertools.chain(*all_data))      # concatenate all received lists
        data          = zlib.decompress(bytes(all_data_flat)) # unzip data
        data          = pickle.loads(data)                    # unpickle data
        log.debug(f'data: {data}')
        if self.process(data): break # start data processing
        self.result_available.set()
      except Exception as ex:
        log.debug(ex)
        self.connect() # try to reconnect
        continue
    self._disconnect()
