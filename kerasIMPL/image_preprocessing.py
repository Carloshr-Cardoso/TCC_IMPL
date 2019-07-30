from os import listdir, path, makedirs
from PIL import Image

input_path = "./data/"
output_path = "./input/"
size = 64

loaded_images = list()
k = 0
x2 = []
if not path.exists(output_path):
   makedirs(output_path)

for filename in listdir(input_path):
    img_data = Image.open(input_path + filename)

    img_data = img_data.resize((size, size))
    img_data.save(output_path + 'simp_faces_'+str(k)+'.png', format='png')
    k += 1
    print('> loaded %s %s' %(filename, img_data.size))
    x2.append(img_data.size[1])
    # if k >= 4938:
    #     break

