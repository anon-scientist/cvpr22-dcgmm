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
from utils             import log
from model             import Model
from metric            import Metric
import tensorflow      as tf
import numpy           as np
from collections       import defaultdict

from .dgr_gen.VAE import VAE
from .dgr_gen.GAN import GAN


'''
VAE needs data normalized [0, +1]
GAN needs data normalized [-1, +1]

Change the code in the TF2_Dataset file
'''


class DGR(Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.replay_ratio   = kwargs['replay_ratio']
    self.balance_type   = kwargs['balance_type'] # FIXME: not used!
    self.rehearsal_type = kwargs['rehearsal_type'] # FIXME: not used!
    self.generator_type = kwargs['generator_type']

    self.drop_model     = kwargs['drop_model']
    self.drop_generator = kwargs['drop_generator']

    self.epsilon     = self.parser.add_argument('--epsilon'    , type=float, default=1e-4       , help='the learning rate')
    self.input_size  = self.parser.add_argument('--input_size' , type=int  , default=[28, 28, 1], help='the input dimensions')
    self.num_classes = self.parser.add_argument('--num_classes', type=int  , default=10         , help='the output dimensions')

    if self.generator_type == 'VAE':
      self.latent_dim        = self.parser.add_argument('--latent_dim'       , type=int  , default=25                          , help='the dimension of the latent vector')
      self.conditional       = self.parser.add_argument('--conditional'      , type=str  , default='no' , choices=['yes', 'no'], help='if the VAE should be conditional based')
      self.beta              = self.parser.add_argument('--beta'             , type=float, default=1.0                         , help='the beta factor for disentangling the VAE')
      self.epsilon_vae       = self.parser.add_argument('--epsilon_vae'      , type=float, default=1e-3                        , help='the learning rate of the VAE')
      self.batch_size_vae    = self.parser.add_argument('--batch_size_vae'   , type=int  , default=100                         , help='the size of mini batches of the VAE') # FIXME: not used!
      self.number_epochs_vae = self.parser.add_argument('--number_epochs_vae', type=int  , default=50                          , help='the number of epochs of the VAE')
      # use some attributes as boolean
      self.conditional = True if self.conditional == 'yes' else False
    elif self.generator_type == 'GAN':
      self.noise_dim         = self.parser.add_argument('--noise_dim'        , type=int  , default=75                          , help='the dimension of the noise vector')
      self.conditional       = self.parser.add_argument('--conditional'      , type=str  , default='no' , choices=['yes', 'no'], help='if the GAN should be conditional based')
      self.wasserstein       = self.parser.add_argument('--wasserstein'      , type=str  , default='no' , choices=['yes', 'no'], help='if the GAN should be wasserstein based')
      self.epsilon_gan       = self.parser.add_argument('--epsilon_gan'      , type=float, default=1e-3                        , help='the learning rate of the GAN')
      self.batch_size_gan    = self.parser.add_argument('--batch_size_gan'   , type=int  , default=100                         , help='the size of mini batches of the GAN') # FIXME: not used!
      self.number_epochs_gan = self.parser.add_argument('--number_epochs_gan', type=int  , default=50                          , help='the number of epochs of the GAN')
      # use some attributes as boolean
      self.conditional = True if self.conditional == 'yes' else False
      self.wasserstein = True if self.wasserstein == 'yes' else False

    # evaluation metrics
    metrics      = ['accuracy_score'] # + ['loss']
    self.metrics = self.parser.add_argument('--metrics', type=str, default=metrics, help='the evaluations metrics')
    self.metric  = Metric(self)

    # ---

    if self.generator_type == 'VAE':
      self.generator = VAE(self.input_size, self.num_classes, self.latent_dim, self.conditional, self.beta, self.epsilon_vae, self.batch_size_vae, self.number_epochs_vae)
    elif self.generator_type == 'GAN':
      self.generator = GAN(self.input_size, self.num_classes, self.noise_dim, self.conditional, self.wasserstein, self.epsilon_gan, self.batch_size_gan, self.number_epochs_gan)

    self.solver = self.build_solver()
    self.loss_solver = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    self.optimizer_solver = tf.keras.optimizers.Adam(self.epsilon)
    #self.optimizer_solver = tf.keras.optimizers.SGD(self.epsilon, momentum=0.9)

    #self.print_summary()

  def build_solver(self):
    return tf.keras.Sequential([
      tf.keras.Input(self.input_size),

      tf.keras.layers.Flatten(),

      tf.keras.layers.Dense(400),
      tf.keras.layers.ReLU(),

      tf.keras.layers.Dense(400),
      tf.keras.layers.ReLU(),

      tf.keras.layers.Dense(400),
      tf.keras.layers.ReLU(),

      tf.keras.layers.Dense(self.num_classes, activation='softmax'),
    ], name='Solver')

  def print_summary(self):
    if self.generator_type == 'VAE':
      self.generator.encoder.summary()
      self.generator.decoder.summary()
    elif self.generator_type == 'GAN':
      self.generator.generator.summary()
      self.generator.discriminator.summary()
    self.solver.summary()

  def reset_generator(self):
    if self.generator_type == 'VAE':
      self.generator.encoder = self.generator.build_encoder()
      self.generator.decoder = self.generator.build_decoder()
    elif self.generator_type == 'GAN':
      self.generator.generator = self.generator.build_generator()
      self.generator.discriminator = self.generator.build_discriminator()

  def reset_model(self):
    self.solver = self.build_solver()

  def build(self, **kwargs):
    log.info('build DGR model')
    return self

  def get_model_variables(self):
    # FIXME: list of dicts is returned instead of single dict
    if self.generator_type == 'VAE':
      return [net.trainable_variables for net in [self.solver, self.generator.encoder, self.generator.decoder]]
    elif self.generator_type == 'GAN':
      return [net.trainable_variables for net in [self.solver, self.generator.generator, self.generator.discriminator]]

  def return_fake_labels(self, fake_data):
    fake_outputs = self.solver(fake_data)
    fake_predictions = tf.argmax(fake_outputs, axis=1)

    return tf.one_hot(fake_predictions, self.num_classes)

  def train_step(self, xs, ys, **kwargs):
    with tf.GradientTape() as sol_tape:
      logits = self.solver(xs)
      loss = self.loss_solver(ys, logits)

    gradients_sol = sol_tape.gradient(loss, self.solver.trainable_variables)
    self.optimizer_solver.apply_gradients(zip(gradients_sol, self.solver.trainable_variables))

  def test_step(self, xs, ys=None, **kwargs):
    logits   = self.solver(xs)
    loss     = self.loss_solver(ys, logits)

    y_pred   = tf.argmax(logits, axis=1)
    y_true   = tf.argmax(ys, axis=1)

    metric_values = self.metric.eval(
      dict    = True  ,
      loss    = loss  ,
      y_true  = y_true,
      y_pred  = y_pred,
      special = {
        'accuracy_score': dict(normalize=True),
        }
    )

    return {'DGR': metric_values}

  def test(self, test_iterator, **kwargs):
    results = defaultdict(list)
    for xs, ys in test_iterator:
      test_results = self.test_step(xs, ys)

      for layer_name, metric_and_values in test_results.items():
        for metric_name, metric_value in metric_and_values.items():
          results[(layer_name, metric_name)] += [metric_value]

    return_results = dict()
    for (layer_name, metric_name), metric_values in results.items():
      if 'accuracy' in metric_name: format_str  = '{:10.1%}'
      else                        : format_str  = '{:10.2f}'
      metric_values                             = np.mean(metric_values)
      return_results[(layer_name, metric_name)] = (metric_values, format_str.format(metric_values))

    return return_results
