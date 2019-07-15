import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy



## Helper Functions

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)

        return f1 * x + f2 * abs(x)

def inverse_transform(images):
    return (images+1.)/2.

def save_images(images, size, image_path):
    return plt.imsave(inverse_transform(images), size, image_path)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h*size[0], w*size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img

## Definindo A Rede Geradora
def generator(z):

    initializer = tf.truncated_normal_initializer(stddev=0.02)
    zP = slim.fully_connected(z, 4*4*256, normalizer_fn=slim.batch_norm,
                              activation_fn=tf.nn.relu, scope='g_project',
                              weights_initializer=initializer)
    zCon = tf.reshape(zP, [-1, 4, 4, 256])

    gen1 = slim.convolution2d_transpose(zCon, num_outputs=64, kernel_size=(5,5), stride=(2,2),
                                        padding='SAME', normalizer_fn=slim.batch_norm,
                                        activation_fn=tf.nn.relu, scope='g_conv1',
                                        weights_initializer=initializer)

    gen2 = slim.convolution2d_transpose(gen1, num_outputs=32, kernel_size=(5,5), stride=(2,2),
                                        padding='SAME', normalizer_fn=slim.batch_norm,
                                        activation_fn=tf.nn.relu, scope='g_conv2',
                                        weights_initializer=initializer)

    gen3 = slim.convolution2d_transpose(gen2, num_outputs=16, kernel_size=(5,5), stride=(2,2),
                                        padding='SAME', normalizer_fn=slim.batch_norm,
                                        activation_fn=tf.nn.relu, scope='g_conv3',
                                        weights_initializer=initializer)

    g_out = slim.convolution2d_transpose(gen3, num_outputs=1, kernel_size=(100,100), padding='SAME',
                                        biases_initializer=None, activation_fn=tf.nn.tanh,
                                        scope='g_out', weights_initializer=initializer)

    return g_out


def discriminator(image, reuse=False):

    """
    Definição da Rede Discriminadora
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    dis1 = slim.convolution2d(image, num_outputs=16, kernel_size=(4,4), stride=(2,2),
                              padding='SAME', biases_initializer=None, activation_fn=lrelu,
                              reuse=reuse, scope='d_conv1', weights_initializer=initializer)

    dis2 = slim.convolution2d(dis1, num_outputs=32, kernel_size=(4,4), stride=(2,2),
                              padding='SAME', normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                              reuse=reuse, scope='d_conv2', weights_initializer=initializer)

    dis3 = slim.convolution2d(dis2, num_outputs=64, kernel_size=(4,4), stride=(2,2),
                              padding='SAME', normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                              reuse=reuse, scope='d_conv3', weights_initializer=initializer)

    d_out = slim.fully_connected(slim.flatten(dis3), num_outputs=1, activation_fn=tf.nn.sigmoid,
                                 reuse=reuse, scope='d_out', weights_initializer=initializer)

    return d_out

## Concatenação das Redes

tf.reset_default_graph()

z_size = 100

z_in = tf.placeholder(shape=(None, z_size), dtype=tf.float32)
real_in = tf.placeholder(shape=(None, 100,100,3), dtype=tf.float32)

Gz = generator(z_in)
Dx = discriminator(real_in)
Dg = discriminator(Gz, reuse=True)

# Otimizadores
d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) # Otimização do Discriminador
g_loss = -tf.reduce_mean(tf.log(Dg)) # Otimização do Gerador

tvars = tf.trainable_variables()

# Gradient Descent
trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
trainerG = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
d_grads = trainerD.compute_gradients(d_loss, tvars[9:])
g_grads = trainerG.compute_gradients(g_loss, tvars[0:9])

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerG.apply_gradients(g_grads)


## Treinamento da GAN

batch_size = 128
epochs = 500000
training_path = './data/new_homer/'
saved_model_path = './models/saved'

init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        zs = np.random.uniform(-1.0,1.0, size=(batch_size, z_size))
        # Training the Network