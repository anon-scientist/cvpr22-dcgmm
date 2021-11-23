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
import tensorflow      as tf
import numpy           as np


class VAE():
  def __init__(self, data_dim, label_dim, latent_dim, conditional, beta, epsilon_vae, batch_size_vae, number_epochs_vae):
    self.data_dim          = data_dim
    self.label_dim         = label_dim
    self.latent_dim        = latent_dim
    self.conditional       = conditional

    self.beta              = beta
    self.epsilon_vae       = epsilon_vae
    self.batch_size_vae    = batch_size_vae # FIXME: not used!
    self.number_epochs_vae = number_epochs_vae

    # ---

    self.encoder = self.build_encoder()
    self.decoder = self.build_decoder()

    self.optimizer = tf.keras.optimizers.Adam(self.epsilon_vae, beta_1=0.05, beta_2=0.95)

    # self.trainable_variables = [*self.encoder.trainable_variables, *self.decoder.trainable_variables] # FIXME: not used!

  def build_encoder(self):
    data_input = tf.keras.Input(self.data_dim)

    if self.conditional:
      conditional_input = tf.keras.Input((self.label_dim,))

      #val = tf.keras.layers.Embedding(self.label_dim, tf.reduce_prod(self.data_dim))(conditional_input)
      val = tf.keras.layers.Dense(tf.reduce_prod(self.data_dim))(conditional_input)
      val = tf.keras.layers.ReLU()(val)

      val = tf.keras.layers.Reshape(self.data_dim)(val)
      val = tf.keras.layers.Concatenate()([data_input, val])
    else:
      val = data_input

    val = tf.keras.layers.Conv2D(32, (5, 5), (2, 2), padding='same')(val) # INFO: alternative kernel_size (3, 3)
    val = tf.keras.layers.ReLU()(val)

    val = tf.keras.layers.Conv2D(64, (5, 5), (2, 2), padding='same')(val) # INFO: alternative kernel_size (3, 3)
    val = tf.keras.layers.ReLU()(val)

    val = tf.keras.layers.Flatten()(val)
    val = tf.keras.layers.Dense(100)(val)
    val = tf.keras.layers.ReLU()(val)

    val = tf.keras.layers.Dense(self.latent_dim)(val)
    val = tf.keras.layers.ReLU()(val)

    output = tf.keras.layers.Dense(self.latent_dim + self.latent_dim)(val)

    if self.conditional:
      return tf.keras.Model(inputs=[data_input, conditional_input], outputs=output, name='VAE_Encoder')
    else:
      return tf.keras.Model(inputs=data_input, outputs=output, name='VAE_Encoder')

  def build_decoder(self):
    latent_input = tf.keras.Input(self.latent_dim)

    if self.conditional:
      conditional_input = tf.keras.Input((self.label_dim,))

      #val = tf.keras.layers.Embedding(self.label_dim, self.latent_dim)(conditional_input)
      val = tf.keras.layers.Dense(self.latent_dim)(conditional_input)
      val = tf.keras.layers.ReLU()(val)

      val = tf.keras.layers.Flatten()(val)
      val = tf.keras.layers.Concatenate()([latent_input, val])
    else:
      val = latent_input

    val = tf.keras.layers.Dense(100)(val)
    val = tf.keras.layers.ReLU()(val)

    val = tf.keras.layers.Dense((self.data_dim[0] // 4) * (self.data_dim[1] // 4) * 64)(val)
    val = tf.keras.layers.ReLU()(val)

    val = tf.keras.layers.Reshape(((self.data_dim[0] // 4), (self.data_dim[1] // 4), 64))(val)
    val = tf.keras.layers.Conv2DTranspose(32, (5, 5), (2, 2), padding='same')(val) # INFO: alternative kernel_size (3, 3)
    val = tf.keras.layers.ReLU()(val)

    output = tf.keras.layers.Conv2DTranspose(1, (5, 5), (2, 2), padding='same', activation='sigmoid')(val) # INFO: alternative kernel_size (3, 3)

    if self.conditional:
      return tf.keras.Model(inputs=[latent_input, conditional_input], outputs=output, name='VAE_Decoder')
    else:
      return tf.keras.Model(inputs=latent_input, outputs=output, name='VAE_Decoder')

  def encode(self, xs, ys):
    if self.conditional:
      mean, logvar = tf.split(self.encoder([xs, ys]), 2, axis=1)
    else:
      mean, logvar = tf.split(self.encoder(xs), 2, axis=1)
    return mean, logvar

  def reparameterize(self, means, logvars):
    eps = tf.random.normal(tf.shape(means))
    sig = tf.exp(0.5 * logvars)
    return means + eps * sig

  def decode(self, zs, ys):
    if self.conditional:
      reconstructed = self.decoder([zs, ys])
    else:
      reconstructed = self.decoder(zs)
    return reconstructed

  def compute_loss(self, real_data, real_labels):
    means, logvars = self.encode(real_data, real_labels)
    zs = self.reparameterize(means, logvars)
    reconstructed = self.decode(zs, real_labels)

    reconstructed_loss = tf.keras.losses.binary_crossentropy(real_data, reconstructed)
    reconstructed_loss = tf.reduce_mean(tf.reduce_sum(reconstructed_loss, axis=(1, 2)))
    kl_loss = -0.5 * (1 + logvars - tf.square(means) - tf.exp(logvars))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

    return reconstructed_loss + self.beta * kl_loss

  def train_step(self, real_data, real_labels=None):
    # FIXME: use only one tape for both -> one model, but splitted into two separate ones
    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
      total_loss = self.compute_loss(real_data, real_labels)

      gradients_encoder = enc_tape.gradient(total_loss, self.encoder.trainable_variables)
      gradients_decoder = dec_tape.gradient(total_loss, self.decoder.trainable_variables)

    self.optimizer.apply_gradients(zip(gradients_encoder, self.encoder.trainable_variables))
    self.optimizer.apply_gradients(zip(gradients_decoder, self.decoder.trainable_variables))

  def return_fake_data(self, number, labels=None):
    eps = tf.random.normal([number, self.latent_dim])
    return self.decode(eps, labels)
