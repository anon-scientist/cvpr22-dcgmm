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


class GAN():
  def __init__(self, data_dim, label_dim, noise_dim, conditional, wasserstein, epsilon_gan, batch_size_gan, number_epochs_gan):
    self.data_dim          = data_dim
    self.label_dim         = label_dim
    self.noise_dim         = noise_dim
    self.conditional       = conditional

    self.wasserstein       = wasserstein
    self.epsilon_gan       = epsilon_gan
    self.batch_size_gan    = batch_size_gan # FIXME: not used!
    self.number_epochs_gan = number_epochs_gan

    self.gp_weight       = 10 # FIXME: fixed initialization!
    self.gen_iterations  = 1  # FIXME: fixed initialization!
    self.disc_iterations = 3  # FIXME: fixed initialization!

    # ---

    self.generator = self.build_generator()
    self.discriminator = self.build_discriminator()

    self.loss_generator = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    self.loss_discriminator = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    self.optimizer_generator = tf.keras.optimizers.Adam(self.epsilon_gan, beta_1=0.05, beta_2=0.95) # INFO: Wasserstein 0.5 and 0.9
    self.optimizer_discriminator = tf.keras.optimizers.Adam(self.epsilon_gan, beta_1=0.05, beta_2=0.95) # INFO: Wasserstein 0.5 and 0.9

  def build_generator(self):
    noise_input = tf.keras.Input(self.noise_dim)

    if self.wasserstein:
      return tf.keras.Sequential([
        noise_input,

        tf.keras.layers.Dense(4 * 4 * 256, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Reshape((4, 4, 256)),

        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), (1, 1), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), (1, 1), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(1, (3, 3), (1, 1), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('tanh'),

        # FIXME: only if dims are not 2 based and smaller than 32
        tf.keras.layers.Cropping2D((2, 2))
      ], name='WGAN-GP_Generator')
    else:
      if self.conditional:
        conditional_input = tf.keras.Input((self.label_dim,))

        #val = tf.keras.layers.Embedding(self.label_dim, tf.reduce_prod(self.noise_dim))(conditional_input)
        val = tf.keras.layers.Dense(tf.reduce_prod(self.noise_dim))(conditional_input)
        val = tf.keras.layers.BatchNormalization()(val)
        #val = tf.keras.layers.ReLU()(val)
        val = tf.keras.layers.LeakyReLU(0.2)(val)
        #val = tf.keras.layers.Dropout(0.3)(val)

        val = tf.keras.layers.Flatten()(val)
        val = tf.keras.layers.Concatenate()([noise_input, val])
      else:
        val = noise_input

      val = tf.keras.layers.Dense(1024)(val)
      val = tf.keras.layers.BatchNormalization()(val)
      #val = tf.keras.layers.ReLU()(val)
      val = tf.keras.layers.LeakyReLU(0.2)(val)
      #val = tf.keras.layers.Dropout(0.3)(val)

      val = tf.keras.layers.Dense((self.data_dim[0] // 4) * (self.data_dim[1] // 4) * 128)(val)
      val = tf.keras.layers.BatchNormalization()(val)
      #val = tf.keras.layers.ReLU()(val)
      val = tf.keras.layers.LeakyReLU(0.2)(val)
      #val = tf.keras.layers.Dropout(0.3)(val)

      val = tf.keras.layers.Reshape(((self.data_dim[0] // 4), (self.data_dim[1] // 4), 128))(val)

      val = tf.keras.layers.Conv2DTranspose(64, (4, 4), (2, 2), padding='same')(val)
      val = tf.keras.layers.BatchNormalization()(val)
      #val = tf.keras.layers.ReLU()(val)
      val = tf.keras.layers.LeakyReLU(0.2)(val)
      #val = tf.keras.layers.Dropout(0.3)(val)

      output = tf.keras.layers.Conv2DTranspose(1, (4, 4), (2, 2), padding='same', activation='tanh')(val)

      if self.conditional:
        return tf.keras.Model(inputs=[noise_input, conditional_input], outputs=output, name='CGAN_Generator')
      else:
        return tf.keras.Model(inputs=noise_input, outputs=output, name='GAN_Generator')

  def build_discriminator(self):
    data_input = tf.keras.Input(self.data_dim)

    if self.wasserstein:
      return tf.keras.Sequential([
        data_input,

        # FIXME: only if dims are not 2 based and smaller than 32
        tf.keras.layers.ZeroPadding2D((2, 2)),

        tf.keras.layers.Conv2D(64, (5, 5), (2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2D(128, (5, 5), (2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(256, (5, 5), (2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(512, (5, 5), (2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
      ], name='WGAN-GP_Discriminator')
    else:
      if self.conditional:
        conditional_input = tf.keras.Input((self.label_dim,))

        #val = tf.keras.layers.Embedding(self.label_dim, tf.reduce_prod(self.data_dim))(conditional_input)
        val = tf.keras.layers.Dense(tf.reduce_prod(self.data_dim))(conditional_input)
        #val = tf.keras.layers.BatchNormalization()(val)
        #val = tf.keras.layers.ReLU()(val)
        val = tf.keras.layers.LeakyReLU(0.2)(val)
        val = tf.keras.layers.Dropout(0.3)(val)

        val = tf.keras.layers.Reshape(self.data_dim)(val)
        val = tf.keras.layers.Concatenate()([data_input, val])
      else:
        val = data_input

      val = tf.keras.layers.Conv2D(64, (4, 4), (2, 2), padding='same')(val)
      #val = tf.keras.layers.BatchNormalization()(val)
      #val = tf.keras.layers.ReLU()(val)
      val = tf.keras.layers.LeakyReLU(0.2)(val)
      val = tf.keras.layers.Dropout(0.3)(val)

      val = tf.keras.layers.Conv2D(128, (4, 4), (2, 2), padding='same')(val)
      #val = tf.keras.layers.BatchNormalization()(val)
      #val = tf.keras.layers.ReLU()(val)
      val = tf.keras.layers.LeakyReLU(0.2)(val)
      val = tf.keras.layers.Dropout(0.3)(val)

      val = tf.keras.layers.Flatten()(val)

      val = tf.keras.layers.Dense(1024)(val)
      #val = tf.keras.layers.BatchNormalization()(val)
      #val = tf.keras.layers.ReLU()(val)
      val = tf.keras.layers.LeakyReLU(0.2)(val)
      val = tf.keras.layers.Dropout(0.3)(val)

      output = tf.keras.layers.Dense(1, activation='sigmoid')(val)

      if self.conditional:
        return tf.keras.Model(inputs=[data_input, conditional_input], outputs=output, name='CGAN_Discriminator')
      else:
        return tf.keras.Model(inputs=data_input, outputs=output, name='GAN_Discriminator')

  def generate(self, zs, ys):
    if self.conditional:
      return self.generator([zs, ys])
    else:
      return self.generator(zs)

  def discriminate(self, xs, ys):
    if self.conditional:
      return self.discriminator([xs, ys])
    else:
      return self.discriminator(xs)

  def calculate_generator_loss(self, logits):
    if self.wasserstein:
      return -tf.reduce_mean(logits)
    else:
      return self.loss_generator(tf.ones_like(logits), logits)

  def calculate_discriminator_loss(self, real_logits, fake_logits):
    if self.wasserstein:
      return tf.subtract(
        tf.reduce_mean(fake_logits),
        tf.reduce_mean(real_logits)
      )
    else:
      return tf.add(
        self.loss_discriminator(tf.ones_like(real_logits), real_logits),
        self.loss_discriminator(tf.zeros_like(fake_logits), fake_logits)
      )

  def gradient_penalty(self, real_images, fake_images):
    alpha = tf.random.normal([tf.shape(real_images)[0], 1, 1, 1], 0., 1.)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
      gp_tape.watch(interpolated)
      pred = self.discriminator(interpolated)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))

    return tf.reduce_mean((norm - 1.0) ** 2)

  def train_step(self, real_data, real_labels=None):
    if self.wasserstein:
      for _ in range(self.disc_iterations):
        noise = tf.random.normal([real_data.shape[0], self.noise_dim])

        with tf.GradientTape() as dis_tape:
          fake_data = self.generator(noise)

          real_output = self.discriminator(real_data)
          fake_output = self.discriminator(fake_data)

          dis_loss = self.calculate_discriminator_loss(real_output, fake_output)
          dis_loss += self.gradient_penalty(real_data, fake_data) * self.gp_weight

          gradients_of_dis = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)

        self.optimizer_discriminator.apply_gradients(zip(gradients_of_dis, self.discriminator.trainable_variables))

      for _ in range(self.gen_iterations):
        noise = tf.random.normal([real_data.shape[0], self.noise_dim])

        with tf.GradientTape() as gen_tape:
          gen_data = self.generator(noise)

          gen_output = self.discriminator(gen_data)

          gen_loss = self.calculate_generator_loss(gen_output)
          gradients_of_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

        self.optimizer_generator.apply_gradients(zip(gradients_of_gen, self.generator.trainable_variables))
    else:
      noise = tf.random.normal([real_data.shape[0], self.noise_dim])

      with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        fake_data = self.generate(noise, real_labels)

        real_output = self.discriminate(real_data, real_labels)
        fake_output = self.discriminate(fake_data, real_labels)

        gen_loss = self.calculate_generator_loss(fake_output)
        dis_loss = self.calculate_discriminator_loss(real_output, fake_output)

        gradients_of_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_dis = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)

      self.optimizer_generator.apply_gradients(zip(gradients_of_gen, self.generator.trainable_variables))
      self.optimizer_discriminator.apply_gradients(zip(gradients_of_dis, self.discriminator.trainable_variables))

  def return_fake_data(self, number, labels=None):
    noise = tf.random.normal([number, self.noise_dim])
    return self.generate(noise, labels)
