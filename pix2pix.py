# coding:utf-8

import numpy as np

import h5py
import matplotlib.pyplot as plt

import keras.backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam,SGD

from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D


# このあたりは引数に設定する
dataset_path = "./data/*.hdf5"
patch_size = 32
batch_size = 12
epoch      = 1000

# 学習データの読み込み

def normalization(X):
    return X / 127.5 - 1

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


def conv_block_unet(x, f, name, bn_axis, bn=True, strides=(2,2)):
    x = LeakyReLU(0.2)(x)
    x = Conv2D(f, (3,3), strides=strides, name=name, padding='same')(x)
    if bn: x = BatchNormalization(axis=bn_axis)(x)
    return x

def up_conv_block_unet(x, x2, f, name, bn_axis, bn=True, dropout=False):
    x = Activation('relu')(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(f, (3,3), name=name, padding='same')(x)
    if bn: x = BatchNormalization(axis=bn_axis)(x)
    if dropout: x = Dropout(0.5)(x)
    x = Concatenate(axis=bn_axis)([x, x2])
    return x

# Generator
def generator_unet_upsampling(img_shape, disc_img_shape, model_name="generator_unet_upsampling"):

    filters_num = 64
    axis_num = -1
    channels_num = img_shape[-1]
    min_s = min(img_shape[:-1])

    unet_input = Input(shape=img_shape, name="unet_input")

    conv_num = int(np.floor(np.log(min_s)/np.log(2)))
    list_filters_num = [filters_num*min(8, (2**i)) for i in range(conv_num)]

    # Encoder 入力をちっちゃく
    first_conv = Conv2D(list_filters_num[0], (3,3), strides=(2,2), name='unet_conv2D_1', padding='same')(unet_input)
    list_encoder = [first_conv]
    for i, f in enumerate(list_filters_num[1:]):
        name = 'unet_conv2D_' + str(i+2)
        conv = conv_block_unet(list_encoder[-1], f, name, axis_num)
        list_encoder.append(conv)

    # prepare decode filters
    list_filters_num = list_filters_num[:2][::-1]
    if len(list_filters_num) < conv_num-1: 
        list_filters_num.append(filters_num)
    
    # Decoder
    fist_up_conv = up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                        list_filters_num[0], "unet_upconv2D_1", axis_num, dropout=True)
    list_decoder = [first_up_conv]
    for i, f in enumerate(list_filters_num[1:]):
        name = "unet_upconv2D" + str(i+2)
        if i<2:
            d = True
        else:
            d = False
        up_conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i+3)], f,
                        name, axis_num, dropout=d, list_decoder.append(up_conv))
    
    x = Activation('relu')(list_decoder[-1])
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(disc_img_shape[-1], (3,3), name="last_conv", padding='same')(x)
    x = Activation('tanh')(x)

    generator_unet = Model(inputs=[unet_input], outputs=[x])
    return generator_unet
       
# Discriminator
def DCGAN_discriminator(img_shape, disc_img_shape, patch_num, model_name='discriminator'):
    disc_raw_img_shape = (disc_img_shape[0], disc_img_shape=[1], img_shape=[-1])
    list_input = [Input(shape=disc_img_shape, name='dist_input'+str(i)) for i in range(patch_num)]
    list_raw_input = [Input(shape=disc_raw_img_shape, name='disc_raw_input'+str(i))]

    axis_num = -1
    filters_num = 64
    conv_num = int(np.floor(np.log(disc_img_shape[1]/np.log(2))))
    list_filters = [filters_num*min(8, (2**i)) for i in range(conv_num)]

    # Fist Conv
    generated_patch_input = Input(shape=disc_img_shape, name='discriminator_inpit')
    xg = Conv2D(list_filters[0], (3,3), strides=(2,2), name='raw_disc_conv2d_1', padding='same')(raw_patch_input)
    xg = BatchNormalization(axis=axis_num)(xg)
    xg = LeakyReLU(0.2)(xg)

    # Firt Raw Conv
    raw_patch_input = Input(shape=disc_img_shape, name='discriminator_raw_input')
    xr = Conv2D(list_filters[0], (3,3), strides=(2,2), name='raw_disc_conv2d_1', padding='same')(raw_patch_input)
    xr = BatchNormalization(axis=axis_num)(xr)
    xr = LeakyReLU(0.2)(xr)

    # Next Conv
    for i, f in enumerate(list_filters[1:]):
        name = 'disc_conv2d_' + str(i+2)
        x = Concatenate(axis=axis_num)([xg, xr)
        x = Conv2D(f, (3,3), strides=(2,2), name=name, padding='same')(x)
        x = BatchNormalization(axis=axis_num)(x)
        x = LeakyReLU(0.2)(x)
    
    x_flat = Flatten()(x)
    x = Dense(2, activation='softmax', name='disc_dense')(x_flat)

    Patch_GUN = Model(inputs=[generated_patch_input, raw_patch_input], outputs=[x], name='PatchGAN')

    x = [Patch_GUN([list_input[i], list_raw_input[i]]) for i in range(patch_num)]

    if len(x)1:
        x = Concatenate(axis=axis_num)(x)
    else:
        x = x[0]

    x_out = Dense(2, activation='softmax', name='disc_output')(x)

    discriminator_model = Model(inputs=(list_input+list_raw_input), outputs=[x_out], name=model_name)

    return discriminator_model

def DCGAN(generator, discriimnator, img_shape, patch_size):
    raw_input = Input(shape=img_shape, name='DCGAN_input')
    generated_image = generator(raw_input)

    h, w = img_shape[:-1]
    ph, pw =  patch_size, patch_size

    list_row_idx = [ ( i*ph, (i+1)*ph) for i in range(h//ph)]
    list_col_idx = [ ( i*ph, (i+1)*ph) for i in range(w//ph)]

    list_gen_patch = []
    list_raw_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_row_idx:
            raw_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(raw_input)
            list_raw_patch.append(raw_patch)
            x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(generated_image)
            list_gen_patch.append(x_patch)

    DCGAN_output = discriimnator(list_gen_patch+list_raw_patch)

    DCGAN = Model(inputs=[raw_input],   
                outputs=[generated_image, DCGAN_output],
                name='DCGAN')

    return DCGAN

def load_generator(img_shape, disc_img_shape):
    model = generator_unet_upsampling(img_shape, disc_img_shape)
    return model

def load_DCGAN_discriminator(img_shape, disc_img_shape, patch_num):
    model = DCGAN_discriminator(img_shape, disc_img_shape, patch_num)
    return model

def load_DCGAN(generator, discriminator, img_shape, patch_size):
    model = DCGAN(generator, discriminator, img_shape, patch_size)
    return model

# L1正則化
def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred, - y_true), axis=-1)

# 正規化(-1~1 → 0~1)
def inverse_normalization(X):
    return (X + 1.) / 2.

def to3d(X):
    if X.shap[-1] == 3: return X
    b = X.transpose(3,1,2,0)
    c = np.array(b[0], b[0], b[0])
    return c.transpose(3,1,2,0)

def plot_generated_batch(X_proc, X_raw, generator_model, batch_size, suffix):
    X_gen  = generator_model.predict(X_raw)
    X_raw  = inverse_normalization(X_raw)
    X_peoc = inverse_normalization(X_proc)
    X_gen  = inverse_normalization(X_gen)

    Xs = to3d(X_raw[:5])
    Xg = to3d(X_gen[:5])
    Xr = to3d(X_proc[:5])
    Xs = np.concatenate(Xs, axis=1)
    Xg = np.concatenate(Xg, axis=1)
    Xr = np.concatenate(Xr, axis=1)
    XX = np.concatenate((Xs, Xg, Xr), axis=0)

    plt.imshow(XX)
    plt.axis('off')
    plt.savefig("current_batch_"+suffix+".png")
    plt.clf()
    plt.close()

def extract_patched(X, patch_size):
    list_X = []
    list_row_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[1]// patch_size)]
    list_col_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[2]// patch_size)]
    for col_idx in list_col_idx:
        list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    return list_X

def get_disc_batch(procImage, rawImage, generator_model, batch_counter, patch_size):
    if batch_counter % 2 == 0:
        # produce an output
        X_disc = generator_model.predict(rawImage)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.unit8)
        y_disc[:, 0] = 1

    else:
        X_disc = proImage
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.unit8)

    X_disc = extract_patched(X_disc, patch_size)
    return X_disc, y_disc
        
def train():
    # load data
    rawImage, procImage, rawImage_val, procImage_val = load_data(datasetpath)

    img_shape = rawImage.shape[-3:]
    patch_num = (img_shape[0]// patch_size) * (img_shape[1] // patch_size)
    disc_img_shape = (patch_size, patch_size, procImage.shape[-1])

    # train
    opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_discriminator = Adam(lr=1E3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # load generator model
    generator_mdoel = load_generator(img_shape, disc_img_shape)
    # load discriminator model
    discriminator_model = load_DCGAN_discriminator(img_shape, disc_img_shape, patch_num)

    genetator_model.compile(loss='mae', optimizer=opt_discriminator)
    discriminator_model.trainable = False

    DCGAN_model = load_DCGAN(generator_model, discriminator_model, img_shape, patch_size)

    loss = [l1_loss,'binary_crossentropy']
    loss_weights = [1E1, 1]
    DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

    discriminator_model.trainable = True
    discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

    # start training
    print('start traing')
    for e in range(epoch):

        perm = np.random.permutation(raeImage.shape[0])
        X_procImage = procImage[perm]
        X_rawImage  = rawImage[perm]
        X_procImageIter = [X_procImage[i:i+batch_size] for i in range(0, rawImage.shape[0], batch_size)]
        X_rawImageIter  = [X_rawImage[i:i+batch_size] for i in range(0, rawImage.shape[0], batch_size)]
        b_it = 0
        progbar = generic_utils.Progbar(len(X_procImageIter)*batch_size)
        
        for (X_proc_batch, X_raw_batch in zip(X_procImageIter, X_rawImageIter)):
            b_it += 1
            X_disc, y_disc = get_disc_batch(X_proc_batch, X_raw_batch, generator_model, b_it, patch_size)
            raw_disc, _ = get_disc_batch(X_raw_batch, X_raw_batch, generator_mdoel, 1, patch_size)
            x_disc = X_disc + raw_disc
            # update the discriminator
            disc_loss = discriminator_model.train_on_batch(x_disc, y_disc)

            # create a batch to feed the generator model
            idx = np.random.choice(procImage.shape[0], batch_size)
            X_gen_target, X_gen = procImage[idx], rawImage[idx]
            y_gen = np.zeros((X_gen.shape[0],2), dtype=np.unit8)
            y_gen[:, 1] = 1

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
            # Unfreeze the discriminator
            discriminato_model.trainable = True

            progbar.add(batch_size, values=[
                ("D logloss", disc_loss),
                ("G tot", gen_loss[0]),
                ("G L1", gen_loss[1]),
                ("G logloss", gen_loss[2])
            ])

            # save images for Visualization
            if b_it % (procImage.shape[0]//batch_size//2) == 0:
                plot_generated_batch(X_proc_batch, X_raw_batch, generator_model, batch_size, "training")
                idx = np.random.choice(procImage_val.shape[0], batchsize)
                X_gen_target, X_gen = procImage_val[idx], rawImage_val[idx]
                plot_generated_batch(X_gen_target, X_gen, generator_model, batch_size, "validation")

        print("")
        print('Epoch %s %s, Time: %s' % (e + 1, epoch))
        


            

