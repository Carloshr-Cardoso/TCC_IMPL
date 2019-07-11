import helper
from glob import glob
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

# Importação das Imagens

data_dir = 'dolphin'
data_path = './'+data_dir+'/*.jpg'
data_files = glob(data_path)
print(len(data_files))

# Pre Processamento das Imagens

HEIGHT = 28
WIDTH = 28
shape = (len(data_files), HEIGHT, WIDTH, 3) #Formato(Quantidade de Imagens, Altura, Largura, Canais de Cores)

def get_imagem(path, width, height, padrao_cor):
    # Lendo imagem do Caminho especificado
    image = Image.open(path)

    if image.size != (width, height):
        face_width = face_height = 300
        j = (image.size[0] - width)
        i = (image.size[1] - height)
        image = image.crop([j, i, j+face_width, i+face_height])
        image = image.resize([width, height], Image.BILINEAR)

    return np.array(image.convert(padrao_cor))

def get_batch(files, width, height, padrao_cor='RGB'):
    data_batch = np.array([get_imagem(sample_file, width, height, padrao_cor) for sample_file in files]).astype(np.float32)

    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch

def get_batches(batch_size):
    IMAGE_MAX_VALUE = 255

    index = 0

    while index + batch_size <= shape[0]:
        data_batch = get_batch(data_files[index:index + batch_size], *shape[1:3])

        index += batch_size
        yield data_batch / IMAGE_MAX_VALUE - 0.5


test_images = get_batch(data_files[:10], 64, 64)

plt.imshow(helper.images_square_grid(test_images))