# coding:utf-8


from pathlib import Path
import numpy as np

from keras.preprocessing.image import load_img, img_to_array

dataset_path = Path.cwd() / 'data'
noise_image_dir = dataset_path / 'noise'
truth_image_dir = dataset_path / 'trurh'

h = 64
w = 64

noise_image_path = noise_image_dir.glob('*')
truth_image_path = truth_image_dir.glob('*')

noise_data = []
truth_data = []


print('load noise image')
for noise_image in noise_image_path:
    print('load' + noise_image)
    img = load_img(noise_image, target_size=(h,w))
    imgarray = img_to_array(img)

    noise_data.append(imgarray)
    
print('load truth image')
for truth_image in truth_image_path:
    print('load' + truth_imag)
    img = load_img(truth_image, arget_size=(h,w))
    imgarray = img_to_array(img)

    truth_data.append(imgarray)