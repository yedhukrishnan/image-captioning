import numpy as np
import imageio
import matplotlib.pyplot as plt
import glob

# Get captions
captions_data = open('data/info.txt').read().split('\n')
captions = dict([line.split(',')[:2] for line in captions_data[:-1]])

images = glob.glob('data/resized/*.jpg')

x_data = list()
y_data = list()

for image_file in images:
    print(image_file)
    image = imageio.imread(image_file)
    if image.shape == (250, 250, 3):
        x_data.append(image)
        y_data.append(captions[image_file.split('/')[-1].split('.')[0]])

np.save('data/x_data.npy', np.array(x_data))
np.save('data/y_data.npy', np.array(y_data))
