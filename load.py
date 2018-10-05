# coding: utf-8

"""
データのロードと整形を行うクラス

"""

import numpy as np
import os, sys

from keras.preprocessing.image import load_img, img_to_array

class Load_Image():

    def __init__(self, img_size, dataset_path):

        dataset_path = '/Users/yuuki/Desktop/python_code/datasets/' 
        truth_train  = []
        noise_train  = []
        truth_val    = []
        noise_val    = []
        img_size     = 32


    def normalization(X):
        return X / 127.5 - 1

    def load(self):
        # ノイズ除去画像(学習用)
        truth_train_path = glob.glob(self.dataset_path+'truth_train/*.png')
        for img_path in truth_train_path:
            print ("image load: img_path)
            img = load_img(img_path, target_size = (self.img_size, self.img_size))
            imgarray = img_to_array(img)
            self.truth_train.append(imgarray)
        self.truth_train = self.normalization(truth_train)
        
        # ノイズ画像(学習用)
        noise_train_path = glob.glob(self.dataset_path+'noise_train/*.png')
        for img_path in noise_train_path:
            print ("image load: img_path)
            img = load_img(img_path, target_size = (self.img_size, self.img_size))
            imgarray = img_to_array(img)
            self.noise_train.append(imgarray)
        self.noise_train = self.normalization(noise_train)

        # ノイズ除去画像(テスト用)
        truth_val_path = glob.glob(self.dataset_path+'noise_train/*.png')
        for img_path in truth_val_path:
            print ("image load: img_path)
            img = load_img(img_path, target_size = (self.img_size, self.img_size))
            imgarray = img_to_array(img)
            self.truth_val.append(imgarray)
        self.truth_val = self.normalization(truth_val)

        # ノイズ画像(テスト用)
        noise_val_path = glob.glob(self.dataset_path+'noise_train/*.png')
        for img_path in noise_train_path:
            print ("image load: img_path)
            img = load_img(img_path, target_size = (self.img_size, self.img_size))
            imgarray = img_to_array(img)
            self.noise_val.append(imgarray)
        self.noise_val = self.normalization(noise_val)

        return self.truth_train, self.noise_train, self.truth_val, self.noise_val

def load_data(dataset_path):
    """ 学習データの読み込み
        フォルダごとに取り出して，0~255 → -1~1 に正規化 """
    with h5py.File(dataset_path, "r") as hf:
        # ノイズ画像(学習用)
        noise_train = hf[""][:].astype(np.float32)
        noise_train = normalization(noise_train)

        # ノイズ除去画像(学習用)  
        truth_train = hf[""][:].astype(np.float32)
        truth_train = normalization(noise_train)

        # ノイズ画像(テスト用)
        noise_val = hf[""][:].astype(np.float32)
        noise_val = normalization(noise_train)

        # ノイズ除去画像(テスト用)
        noise_val = hf[""][:].astype(np.float32)
        noise_val = normalization(noise_train)
