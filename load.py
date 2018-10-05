# coding: utf-8

"""
データのロードと整形を行うクラス

"""

import numpy as np
import os, sys
import glob

from keras.preprocessing.image import load_img, img_to_array

class Load_Image():

    def __init__(self, dataset_path, img_size=32):

        self.dataset_path = dataset_path 
        self.truth_train  = []
        self.noise_train  = []
        self.truth_val    = []
        self.noise_val    = []
        self.img_size     = img_size

    def normalization(self, X):
        return X / 127.5 - 1 

    def load(self):
        # ノイズ除去画像(学習用)
        print (self.dataset_path)
        print("truth trian image loading...")
        truth_train_path = glob.glob(self.dataset_path+'truth_train/*.png')
        for img_path in truth_train_path:
            #print ("image load:" + img_path)
            img = load_img(img_path, target_size = (self.img_size, self.img_size))
            imgarray = img_to_array(img)
            self.truth_train.append(imgarray)
        self.truth_train = np.array(self.truth_train).astype(np.float32)
        self.truth_train = self.normalization(self.truth_train)
        
        # ノイズ画像(学習用)
        print("noise train image loading...")
        noise_train_path = glob.glob(self.dataset_path+'noise_train/*.png')
        for img_path in noise_train_path:
            #print ("image load:" + img_path)
            img = load_img(img_path, target_size = (self.img_size, self.img_size))
            imgarray = img_to_array(img)
            self.noise_train.append(imgarray)
        self.noise_train = np.array(self.noise_train).astype(np.float32)
        self.noise_train = self.normalization(self.noise_train)

        # ノイズ除去画像(テスト用)
        print ("truth validation image loading...")
        truth_val_path = glob.glob(self.dataset_path+'truth_val/*.png')
        for img_path in truth_val_path:
            #print ("image load:" + img_path)
            img = load_img(img_path, target_size = (self.img_size, self.img_size))
            imgarray = img_to_array(img)
            self.truth_val.append(imgarray)
        self.truth_val = np.array(self.truth_val).astype(np.float32)
        self.truth_val = self.normalization(self.truth_val)

        # ノイズ画像(テスト用)
        print ("noise validation image loading...")
        noise_val_path = glob.glob(self.dataset_path+'noise_val/*.png')
        for img_path in noise_val_path:
            #print ("image load:" + img_path)
            img = load_img(img_path, target_size = (self.img_size, self.img_size))
            imgarray = img_to_array(img)
            self.noise_val.append(imgarray)
        self.noise_val = np.array(self.noise_val).astype(np.float32)
        self.noise_val = self.normalization(self.noise_val)

        return self.noise_train, self.truth_train, self.noise_val, self.truth_val