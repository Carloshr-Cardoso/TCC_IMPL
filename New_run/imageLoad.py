# from PIL import Image
# from matplotlib import image, pyplot as plt
# import numpy as np
# from os import listdir
import tensorflow as tf
import os


def process_data(path, batch_size):

    HEIGHT, WIDTH, CHANNEL = 100, 100, 3

    images = []
    for each in os.listdir(path):
        images.append(os.path.join(path, each))

    dataset = tf.convert_to_tensor(value=images, dtype=tf.string)

    images_queue = tf.data.Dataset.from_tensors(dataset)
    content = tf.read_file(images_queue)

    # images_queue = tf.train.slice_input_producer([dataset])
    # content = tf.read_file(images_queue[0])

    image = tf.image.decode_png(contents=content, channels=CHANNEL)

    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size=size)
    image.set_shape((HEIGHT, WIDTH, CHANNEL))

    image = tf.cast(image, tf.float32)
    image = image / 255.0

    images_batch = tf.train.shuffle_batch([image], batch_size=batch_size,
                                          num_threads=4, capacity=2246,
                                          min_after_dequeue=2245)

    num_images = len(images)

    return images_batch, num_images


path = './data/new_homer/'
batches, qtd_images = process_data(path, 64)