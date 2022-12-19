import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1] # = 784
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_shape = (original_dim, )
latent_dim = 2


def sampling(args):
    # Reparametrization Trick
    z_mean, z_logvar = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=5)  # ε
    return z_mean + tf.keras.backend.exp(0.5 * z_logvar) * epsilon


inputs = tf.keras.layers.Input(shape=input_shape)
x1 = tf.keras.layers.Dense(256, activation='relu')(inputs)
x2 = tf.keras.layers.Dense(64, activation='relu')(x1)
z_mean = tf.keras.layers.Dense(latent_dim)(x2)
z_logvar = tf.keras.layers.Dense(latent_dim)(x2)
z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_logvar])
encoder = tf.keras.models.Model(inputs, [z_mean, z_logvar, z], name='encoder')
encoder.summary()

latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))
x3 = tf.keras.layers.Dense(64, activation='relu')(latent_inputs)
x4 = tf.keras.layers.Dense(256, activation='relu')(x3)
outputs = tf.keras.layers.Dense(original_dim, activation='sigmoid')(x4)
decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')
decoder.summary()


z_output = encoder(inputs)[2]
outputs = decoder(z_output)
vae = tf.keras.models.Model(inputs, outputs, name='variational_autoencoder')

# 損失関数
# Kullback-Leibler Loss
kl_loss = 1 + z_logvar - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_logvar)
kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
kl_loss *= -0.5

# Reconstruction Loss
reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
reconstruction_loss *= original_dim

vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')


vae.fit(x_train,
        epochs=50,
        batch_size=256,
        validation_data=(x_test, None))

# テスト画像を変換
decoded_imgs = vae.predict(x_test)

# テスト画像と変換画像の表示
n = 10
plt.figure(figsize=(10, 2))
for i in range(n):
    # テスト画像を表示
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 変換された画像を表示
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

if __name__ == '__main__':
    print("hello world.")