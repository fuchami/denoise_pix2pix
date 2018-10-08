# coding:utf-8

import numpy as np

import h5py
import matplotlib.pyplot as plt

from keras.utils import generic_utils
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam,SGD
import keras.backend as K

import model
from load import Load_Image

# このあたりは引数に設定する
patch_size = 32
batch_size = 12
epoch      = 1000


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

def plot_generated_batch(X_truth, X_noise, generator_model, batch_size, suffix):
    X_gen  = generator_model.predict(X_noise)
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
    plt.savefig("./images/current_batch_"+suffix+".png")
    plt.clf()
    plt.close()

def extract_patched(X, patch_size):
    list_X = []
    list_row_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[1]// patch_size)]
    list_col_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[2]// patch_size)]
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    return list_X

def get_disc_batch(truthImage, noiseImage, generator_model, batch_counter, patch_size):
    if batch_counter % 2 == 0:
        # produce an output
        X_disc = generator_model.predict(noiseImage)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1

    else:
        X_disc = truthImage
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)

    X_disc = extract_patched(X_disc, patch_size)
    return X_disc, y_disc
        
def train():
    # load data
    load_img = Load_Image('/media/futami/HDD1/DATASET_KINGDOM/denoise_cifar/')
    # 正解画像、入力画像
    truthImage, noiseImage, truthImage_val, noiseImage_val = load_img.load()
    
    print('truthImgae.shape', truthImage.shape)
    print('noiseImage.shape', noiseImage.shape)
    print('truthImage_val', truthImage_val.shape)
    print('noiseImage_val', noiseImage_val.shape)

    img_shape = noiseImage.shape[-3:]
    print('image_shape: ', img_shape)
    patch_num = (img_shape[0]// patch_size) * (img_shape[1] // patch_size)
    disc_img_shape = (patch_size, patch_size, truthImage.shape[-1])

    print('disc_img_shape:' , disc_img_shape)

    # train
    opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_discriminator = Adam(lr=1E3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # load generator model
    generator_model = model.load_generator(img_shape, disc_img_shape)
    plot_model(generator_model, to_file='./images/model/generator.png', show_shapes=True)
    # load discriminator model
    discriminator_model = model.load_DCGAN_discriminator(img_shape, disc_img_shape, patch_num)
    plot_model(discriminator_model, to_file='./images/model/discriminator.png', show_shapes=True)

    generator_model.compile(loss='mae', optimizer=opt_discriminator)
    discriminator_model.trainable = False

    DCGAN_model = model.load_DCGAN(generator_model, discriminator_model, img_shape, patch_size)
    plot_model(DCGAN_model, to_file='./images/model/DCGAN_model.png', show_shapes=True)

    loss = [l1_loss,'binary_crossentropy']
    loss_weights = [1E1, 1]
    DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

    discriminator_model.trainable = True
    discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)


    # start training
    print('start traing')
    for e in range(epoch):

        perm = np.random.permutation(noiseImage.shape[0])
        X_truthImage = truthImage[perm]
        X_noiseImage = noiseImage[perm]
        X_truthImageIter = [X_truthImage[i:i+batch_size] for i in range(0, noiseImage.shape[0], batch_size)]
        X_noiseImageIter  = [X_noiseImage[i:i+batch_size] for i in range(0, noiseImage.shape[0], batch_size)]
        b_it = 0
        progbar = generic_utils.Progbar(len(X_truthImageIter)*batch_size)
        
        for (X_truth_batch, X_noise_batch) in zip(X_truthImageIter, X_noiseImageIter):
            b_it += 1
            X_disc, y_disc = get_disc_batch(X_truth_batch, X_noise_batch, generator_model, b_it, patch_size)
            noise_disc, _ = get_disc_batch(X_noise_batch, X_noise_batch, generator_model, 1, patch_size)
            x_disc = X_disc + noise_disc
            # update the discriminator
            disc_loss = discriminator_model.train_on_batch(x_disc, y_disc)

            # create a batch to feed the generator model
            idx = np.random.choice(truthImage.shape[0], batch_size)
            X_gen_target, X_gen = truthImage[idx], noiseImage[idx]
            y_gen = np.zeros((X_gen.shape[0],2), dtype=np.uint8)
            y_gen[:, 1] = 1

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
            # Unfreeze the discriminator
            discriminator_model.trainable = True

            progbar.add(batch_size, values=[
                ("D logloss", disc_loss),
                ("G tot", gen_loss[0]),
                ("G L1", gen_loss[1]),
                ("G logloss", gen_loss[2])
            ])

            # save images for Visualization
            if b_it % (truthImage.shape[0]//batch_size//2) == 0:
                plot_generated_batch(X_truth_batch, X_noise_batch, generator_model, batch_size, "training")
                idx = np.random.choice(truthImage_val.shape[0], batch_size)
                X_gen_target, X_gen = truthImage_val[idx], noiseImage_val[idx]
                plot_generated_batch(X_gen_target, X_gen, generator_model, batch_size, "validation")

        print("")
        print('Epoch %s %s' % (e + 1, epoch))
        
def main():

    train()

if __name__ == '__main__' :
    main()
