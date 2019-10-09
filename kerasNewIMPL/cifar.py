import time
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.optimizers import Adam
from keras import initializers
from keras.utils import np_utils
from keras import backend as K


from os import path, makedirs
from PIL import Image

samples_path = './samples/'
all_generated_samples_path = samples_path + 'all_generated_samples/'
models_path = './models/'
d_losses_path = './d_loss_plot/'
g_losses_path = './g_loss_plot/'

if not path.exists(samples_path):
    makedirs(samples_path)
if not path.exists(d_losses_path):
    makedirs(d_losses_path)
if not path.exists(g_losses_path):
    makedirs(g_losses_path)
if not path.exists(models_path):
    makedirs(models_path)
if not path.exists(all_generated_samples_path):
    makedirs(all_generated_samples_path)

quant = 0.05
# quant = 1
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_cars = X_train[(y_train == 1).flatten()] # 1 = Carros
np.random.shuffle(X_cars)
X_train = X_cars[:int(len(X_cars)*quant)].copy()
y_train = y_train[(y_train == 1).flatten()]

cars_test = X_test[(y_test == 1).flatten()] # 1 = Carros
np.random.shuffle(cars_test)
X_test = cars_test[:int(len(cars_test)*quant)].copy()
y_test = y_test[(y_test == 1).flatten()]

print(X_train.shape)
print(X_test.shape)

fig = plt.figure(figsize=(8, 3))
for i in range(0, 10):
    plt.subplot(2, 5, 1 + i, xticks=[], yticks=[])
    plt.imshow(X_train[i])

plt.tight_layout()

num_classes = len(np.unique(y_train))
class_names = ['automobile']

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
    X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
    input_shape = (3, 32, 32)
else:
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    input_shape = (32, 32, 3)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

# the generator is using tanh activation, for which we need to preprocess
# the image data into the range between -1 and 1.

X_train = np.float32(X_train)
X_train = (X_train / 255 - 0.5) * 2
X_train = np.clip(X_train, -1, 1)

X_test = np.float32(X_test)
X_test = (X_train / 255 - 0.5) * 2
X_test = np.clip(X_test, -1, 1)

print('X_train reshape:', X_train.shape)
print('X_test reshape:', X_test.shape)

# latent space dimension
latent_dim = 100

init = initializers.RandomNormal(stddev=0.02)

# Generator network
generator = Sequential()

# FC: 2x2x512
generator.add(Dense(2*2*512, input_shape=(latent_dim,), kernel_initializer=init))
generator.add(Reshape((2, 2, 512)))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))

# # Conv 1: 4x4x256
generator.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))

# Conv 2: 8x8x128
generator.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))

# Conv 3: 16x16x64
generator.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))

# Conv 4: 32x32x3
generator.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                              activation='tanh'))

# prints a summary representation of your model
generator.summary()

# imagem shape 32x32x3
img_shape = X_train[0].shape

# Discriminator network
discriminator = Sequential()

# Conv 1: 16x16x64
discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                         input_shape=(img_shape), kernel_initializer=init))
discriminator.add(LeakyReLU(0.2))

# Conv 2:
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))

# Conv 3:
discriminator.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))

# Conv 3:
discriminator.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))

# FC
discriminator.add(Flatten())

# Output
discriminator.add(Dense(1, activation='sigmoid'))

# prints a summary representation of your model
discriminator.summary()

# Optimizer

discriminator.compile(Adam(lr=0.0003, beta_1=0.5), loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

# d_g = discriminador(generador(z))
discriminator.trainable = False

z = Input(shape=(latent_dim,))
img = generator(z)
decision = discriminator(img)
d_g = Model(inputs=z, outputs=decision)

d_g.compile(Adam(lr=0.0004, beta_1=0.5), loss='binary_crossentropy',
            metrics=['binary_accuracy'])

# prints a summary representation of your model
d_g.summary()

epochs = 10000
batch_size = 32
smooth = 0.1

real = np.ones(shape=(batch_size, 1))
fake = np.zeros(shape=(batch_size, 1))

d_loss = []
g_loss = []

now = time.time()
for e in range(epochs + 1):
    time_epoch = time.time()
    for i in range(len(X_train) // batch_size):
        now_epoch = time.time()
        # Train Discriminator weights
        discriminator.trainable = True

        # Real samples
        X_batch = X_train[i * batch_size:(i + 1) * batch_size]
        d_loss_real = discriminator.train_on_batch(x=X_batch,
                                                   y=real * (1 - smooth))

        # Fake Samples
        z = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))
        X_fake = generator.predict_on_batch(z)
        d_loss_fake = discriminator.train_on_batch(x=X_fake, y=fake)

        # Discriminator loss
        d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])

        # Train Generator weights
        discriminator.trainable = False
        g_loss_batch = d_g.train_on_batch(x=z, y=real)

        print(
            'epoch = {epoch}/{total}, batch = {batch}/{total_batch}, d_loss={loss_d:.3f}, g_loss={loss_g:.3f} - Durou={tempo:.2f}'
                .format(epoch=e + 1,
                        total=epochs,
                        batch=i+1,
                        total_batch=(len(X_train) // batch_size),
                        loss_d=d_loss_batch,
                        loss_g=g_loss_batch[0],
                        tempo=time.time() - now_epoch)
        )

    d_loss.append(d_loss_batch)
    g_loss.append(g_loss_batch[0])

    plt.plot(d_loss, label='d_loss', alpha=0.6)
    plt.title("d_loss" + str(e))
    plt.legend()
    plt.savefig(d_losses_path + 'd_loss_' + str(e) + '.png')
    plt.close()

    plt.plot(g_loss, label='g_loss', alpha=0.6)
    plt.title("g_loss" + str(e))
    plt.legend()
    plt.savefig(g_losses_path + 'g_loss_' + str(e) + '.png')
    plt.close()
    # print('epoch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, d_loss[-1], g_loss[-1]), 100 * ' ')
    print(
        'epoch = {epoch}/{total}, d_loss={loss_d:.3f}, g_loss={loss_g:.3f} - Durou={tempo:.2f}'
            .format(epoch=e + 1,
                    total=epochs,
                    loss_d=d_loss[-1],
                    loss_g=g_loss[-1],
                    tempo=time.time() - time_epoch)
    )
    if e % 1 == 0:
        samples = 8
        x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, latent_dim)))
        sample_images = [((sample + 1.0) * 127.5).astype(np.uint8) for sample in x_fake]

        figure, axes = plt.subplots(1, len(sample_images), figsize=(32, 32))
        for index, axis in enumerate(axes):
            axis.axis('off')
            image_array = sample_images[index]
            axis.imshow(image_array)
            image = Image.fromarray(image_array)
            image.save(all_generated_samples_path + "samples_" + str(e) + "_" + str(index) + ".png")
        plt.savefig(samples_path + 'samples_' + str(e) + ".png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # for k in range(samples):
        #     # plt.subplot(2, 5, k + 1, xticks=[], yticks=[])
        #     # plt.imshow(((x_fake[k] + 1) * 127).astype(np.uint8))
        #     plt.imsave(fname=samples_path + "sample_{}_{}.png".format(e, k), arr=((x_fake[k] + 1) * 127).astype(np.uint8))
        #     discriminator.save_weights(models_path + "disc-cifar-model.h5-" + str(e))
        #     d_g.save_weights(models_path + "gan-cifar-model.h5-" + str(e))
        # plt.imsave(fname=samples_path + "sample_{}.png".format(e), arr=(x_fake * 127).astype(np.uint8))
        #plt.tight_layout()
        #plt.show()

print("tempo Total = {:.2f}".format(time.time() - now))
