# coding:utf-8

import numpy as np
import argparse
import subprocess as sp
import os,sys

import h5py
import matplotlib.pyplot as plt

from keras.utils import generic_utils
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam,SGD
import keras.backend as K

import model
from load import Load_Image

# Line API
def send_image(path_to_img, line_notify_token, m):
    line_notify_api = 'https://notify-api.line.me/api/notify'
    sp.getoutput(
        "curl -X POST {} -H 'Authorization: Bearer {}' -F 'message={}' -F 'imageFile=@{}'".format(line_notify_api, line_notify_token, m, path_to_img))
# L1正則化
def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)

# 正規化(-1~1 → 0~1)
def inverse_normalization(X):
    return (X + 1.) / 2.

def inverse_normalization255(X):
    return ((X + 1.)/ 2.)*255

def to3d(X):
    if X.shape[-1] == 3: return X
    b = X.transpose(3,1,2,0)
    c = np.array(b[0], b[0], b[0])
    return c.transpose(3,1,2,0)

def plot_generated_batch(X_truth, X_noise, cae_model, batch_size, suffix):
    X_gen  = cae_model.predict(X_noise)
    X_noise  = inverse_normalization(X_noise)
    X_truth = inverse_normalization(X_truth)
    X_gen  = inverse_normalization(X_gen)

    # 上からノイズ画像　生成画像、　正解画像（真値）
    Xs = to3d(X_noise[:5])
    Xg = to3d(X_gen[:5])
    Xr = to3d(X_truth[:5])
    Xs = np.concatenate(Xs, axis=1)
    Xg = np.concatenate(Xg, axis=1)
    Xr = np.concatenate(Xr, axis=1)

    XX = np.concatenate((Xs, Xg, Xr), axis=0)

    plt.imshow(XX)
    plt.axis('off')
    plt.savefig("./images/CAE_current_batch_"+suffix+".png")
    plt.clf()
    plt.close()

        
def train(args):

    # 各種パラメータ
    batch_size = args.batchsize
    patch_size = args.patchsize
    print(args.imgsize)

    # load data
    load_img = Load_Image(args.datasetpath, args.imgsize)
    # 正解画像、入力画像
    truthImage, noiseImage, truthImage_val, noiseImage_val = load_img.load()
    
    print('truthImgae.shape', truthImage.shape)
    print('noiseImage.shape', noiseImage.shape)
    print('truthImage_val', truthImage_val.shape)
    print('noiseImage_val', noiseImage_val.shape)

    img_shape = noiseImage.shape[-3:]
    print('image_shape: ', img_shape)

    # train
    # load cae model
    cae_model = model.load_CAE(img_shape)
    plot_model(cae_model, to_file='./images/model/CAE.png', show_shapes=True)
    cae_model.compile(loss='binary_crossentropy', optimizer='adam')

    # start training
    print('start traing')
    for e in range(args.epoch):

        perm = np.random.permutation(noiseImage.shape[0])
        X_truthImage = truthImage[perm]
        X_noiseImage = noiseImage[perm]
        X_truthImageIter = [X_truthImage[i:i+batch_size] for i in range(0, noiseImage.shape[0], batch_size)]
        X_noiseImageIter  = [X_noiseImage[i:i+batch_size] for i in range(0, noiseImage.shape[0], batch_size)]
        b_it = 0
        progbar = generic_utils.Progbar(len(X_truthImageIter)*batch_size)
        
        for (X_truth_batch, X_noise_batch) in zip(X_truthImageIter, X_noiseImageIter):
            b_it += 1
            # update the discriminator
            cae_loss = cae_model.train_on_batch(X_truth_batch, X_noise_batch)


            progbar.add(batch_size, values=[
                ("logloss", cae_loss)
            ])

            # save images for Visualization
            if b_it % (truthImage.shape[0]//batch_size//2) == 0:
                plot_generated_batch(X_truth_batch, X_noise_batch, cae_model, batch_size, "training")
                idx = np.random.choice(truthImage_val.shape[0], batch_size)
                X_gen_target, X_gen = truthImage_val[idx], noiseImage_val[idx]
                plot_generated_batch(X_gen_target, X_gen, cae_model, batch_size, "validation")


        print("")
        print('Epoch %s %s' % (e + 1, args.epoch))
        if e % 100 == 0:
            print("send image to LINE massage !")
            send_image("./images/CAE_current_batch_validation.png", args.line_token, 
                        "Epoch: %s, sent a image: CAE_current_batch_validation.png !" % (e) )

    """ model save """
    if not os.path.exists('./saved_model/'):
        os.makedirs('./saved_model/')

    json_string = cae_model.to_json()
    open('./saved_model/cae_model.json', 'w').write(json_string)
    cae_model.save_weights('./saved_model/CAE_weights.h5')

        
def main():
    parser = argparse.ArgumentParser(description='Train Denoise Convolutional Auto Encoder')
    parser.add_argument('--datasetpath', '-d', type=str, default='/media/futami/HDD1/DATASET_KINGDOM/denoise/')
    parser.add_argument('--line_token', '-l', type=str, required=False)
    parser.add_argument('--imgsize', '-s', default=64)
    parser.add_argument('--epoch', default=2000)
    parser.add_argument('--patchsize', default=32)
    parser.add_argument('--batchsize', default=12)

    args = parser.parse_args()

    train(args)

if __name__ == '__main__' :
    main()
