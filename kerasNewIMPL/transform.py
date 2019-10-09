import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from PIL import Image

# input_path = 'C:/Users/carlos.cardoso/Desktop/DEV_ufpa/KerasNewIMPL/samples/'
input_path = 'C:/Users/carlos.cardoso/Desktop/DEV_ufpa/KerasNewIMPL/Teste 01/samples/'
samples_path = './samples/'
all_generated_samples_path = samples_path + 'all_generated_samples/'

img_data = []
for filename in listdir(input_path):
    pix = np.asarray(Image.open(input_path + filename))
    # plt.imsave(samples_path + filename, pix)
    img_data.append(pix)
    # print('> Loaded %s %s' % (filename, img_data[-1].size))

x = []
# for i in range(550):
#     x.append(i)

# print(len(img_data))
samples = 68
k = 1
for i in range(0, len(img_data), samples):
    x_fake = img_data[i:i+samples]
    # sample_images = [sample for sample in x_fake]
    # sample_images = [((sample + 1.0) * 127.5).astype(np.uint8) for sample in x_fake]

    figure, axes = plt.subplots(8, 8, figsize=(32, 32))
    # figure, axes = plt.subplots(1, len(x_fake), figsize=(32, 32))
    axes = axes.flatten()
    for index, axis in enumerate(axes):
        # axis.axis('off')
        image_array = x_fake[index]
        axis.imshow(image_array)
        image = Image.fromarray(image_array)
        image.save(all_generated_samples_path + "samples_" + str(i) + "_" + str(index) + ".png")
    print("Salvando Imagem {}/{}".format(k, int(len(img_data)/samples)))
    plt.savefig(samples_path + 'samples_' + str(i) + ".png", bbox_inches='tight', pad_inches=0)
    plt.close()
    k += 1
