# coding:utf-8

"""
モデルと重みの読み込み
推論を行うスクリプト

"""

import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils
from keras.preprocessing.image import load_img, img_to_array
import argparse
import os,sys
import glob
import matplotlib.pyplot as plt

def normalization(X):
    return X/ 127.5 -1

def img_load(load_path, img_size):
    X = []
    print(load_path)
    print("input image loading...")

    img_path_list = glob.glob(load_path + '/.*')
    for img_path in img_path_list:
        img = load_img(img_path, target_size=(img_size, img_size)) 
        imgarray = img_to_array(img)
        X.append(imgarray)
    
    X = np.array(X).astype(np.float32)
    X = normalization(X)

    return X

def to3d(X):
    if X.shape[-1] == 3: return X
    b = X.transpose(3,1,2,0)
    c = np.array(b[0], b[0], b[0])
    return c.transpose(3,1,2,0)
def main(args):

    # load model structure
    model = model_from_json(open(args.modeljson).read())

    # load model weights
    model.load_weights(args.modelweight, args.loadimgsize)

    model.summary()

    # load image
    load_img = img_load(args.imgpath)

    # predict
    print("let's predict")
    for img in load_img:
        X_gen = model.predict(img)
        X_gen = to3d(X_gen)

        plt.imshow(X_gen)
        plt.savefig(args.savepath + os.path.basename(img))
        plt.clf()
        plt.close()


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeljson', '-m', type=str, default='./generator_model.json')
    parser.add_argument('--modelweight', '-w', type=str, default='./generator_model.h5')
    parser.add_argument('--imgpath', '-p', type=str, require=True)
    parser.add_argument('--savepath', '-s', type=str, require=True)
    parser.add_argument('--loadimgsize', '-l', type=int, default=64)

    args = parser.parse_args()

    main(args)