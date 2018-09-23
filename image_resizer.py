# Resize all the images to a fixed size
# Skip if the resized image already exists

import imageio
from scipy import misc
from matplotlib import pyplot as plt
import os
import glob

# os.mkdir('data/resized')
file_list = glob.glob('data/*.jpg')

for file_name in file_list:
    new_file_name = file_name.split('/')[1]
    print('Resizing ' + file_name)
    try:
        if os.path.isfile('data/resized/' + new_file_name):
            print('Skipped')
            continue
        image = imageio.imread(file_name)
        resized_image = misc.imresize(image, (250, 250, 3))
        print('Saving resized image')
        imageio.imwrite('data/resized/' + new_file_name, resized_image)
    except:
        print('Error: ' + file_name)


