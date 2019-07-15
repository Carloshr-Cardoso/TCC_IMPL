import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

path = './data/new_homer/'

i = 0
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img))
    plt.imshow(img_array)
    plt.show()

    if i == 10:
        break
    i += 1

