# coding:utf-8

"""
データをロードとかするクラス

"""

import numpy as np
import glob
from pathlib import Path

from keras.preprocessing.image import load_img, img_to_array


class Load_image():


    def __init__(self):
        self.dataset_path = Path.cwd() / 'data'

    
    def load(self):
        