from PIL import Image
from matplotlib import image, pyplot as plt
import numpy as np
from os import listdir
# path = './data/helicopter/'
path = './data/homer_simpson/'
loaded_images = list()
k = 0
x2 = []

for filename in listdir(path):
    img_data = Image.open(path + filename)

    face_width = 350
    face_height = 350
    j = (img_data.size[0] - face_width) // 2
    i = (img_data.size[1] - face_height) // 2

    img_data = img_data.crop([j, i, j + face_width, i + face_height])
    img_data = img_data.resize((100, 100))
    # img_data = img_data.resize((100, 100))
    # name = 'helicopter_' + str(i)
    img_data.save('homer_'+str(k)+'.png', format='png')
    k += 1
    print('> loaded %s %s' %(filename, img_data.size))
    x2.append(img_data.size[1])

