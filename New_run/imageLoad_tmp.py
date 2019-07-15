from glob import glob
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import helper

HEIGHT = 100
WIDTH = 100
data_path = './data/new_homer/'
# data_path = './data/helicopter'
data_files = glob(data_path+'/*.png')
shape = len(data_files), WIDTH, HEIGHT

# print(len(data_files))

def get_image(image_path, mode):
    image = Image.open(image_path)
    image = np.array(image.convert(mode))

    return image

def get_batch(image_files, width, height, mode='RGB'):
    data_batch = np.array([get_image(sample_file, mode) for sample_file in image_files])#.astype(np.float32)

    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))
    return data_batch

def get_batches(batch_size):
    IMAGE_MAX_VALUE = 255

    index = 0
    while index + batch_size <= shape[0]:
        data_batch = get_batch(data_files[index:index+batch_size],*shape[1:3])
        index += batch_size

        yield data_batch / IMAGE_MAX_VALUE - 0.5

test_images = get_batch(data_files, 56, 56)



# Definindo as Entradas da Rede