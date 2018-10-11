# coding:utf-8
"""
pix2pixのモデル構築スクリプト

CAEを追加

"""

from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate, MaxPooling2D, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
import numpy as np


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

    # 畳込みの回数は画像サイズに合わせて各自調整が必要
    conv_num = int(np.floor(np.log(min_s)/np.log(2))) -2 
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
    if len(list_filters_num) < conv_num : 
        list_filters_num.append(filters_num)
    
    # Decoder
    first_up_conv = up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                        list_filters_num[0], "unet_upconv2D_1", axis_num, dropout=True)
    list_decoder = [first_up_conv]
    for i, f in enumerate(list_filters_num[1:]):
        name = "unet_upconv2D" + str(i+2)
        if i<2:
            d = True
        else:
            d = False
        up_conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i+3)], f,
                        name, axis_num, dropout=d)
        list_decoder.append(up_conv)
    
    x = Activation('relu')(list_decoder[-1])
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(disc_img_shape[-1], (3,3), name="last_conv", padding='same')(x)
    x = Activation('tanh')(x)

    generator_unet = Model(inputs=[unet_input], outputs=[x])
    return generator_unet

# Discriminator
def DCGAN_discriminator(img_shape, disc_img_shape, patch_num, model_name='DCGAN_discriminator'):
    disc_raw_img_shape = (disc_img_shape[0], disc_img_shape[1], img_shape[-1])
    list_input = [Input(shape=disc_img_shape, name='dist_input'+str(i)) for i in range(patch_num)]
    list_raw_input = [Input(shape=disc_raw_img_shape, name='disc_raw_input'+str(i)) for i in range(patch_num)]

    axis_num = -1
    filters_num = 64 
    conv_num = int(np.floor(np.log(disc_img_shape[1]/np.log(2))))
    list_filters = [filters_num*min(8, (2**i)) for i in range(conv_num)]

    # Fist Conv
    generated_patch_input = Input(shape=disc_img_shape, name='discriminator_inpit')
    xg = Conv2D(list_filters[0], (3,3), strides=(2,2), name='disc_conv2d_1', padding='same')(generated_patch_input)
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
        x = Concatenate(axis=axis_num)([xg, xr])
        x = Conv2D(f, (3,3), strides=(2,2), name=name, padding='same')(x)
        x = BatchNormalization(axis=axis_num)(x)
        x = LeakyReLU(0.2)(x)
    
    x_flat = Flatten()(x)
    x = Dense(2, activation='softmax', name='disc_dense')(x_flat)

    Patch_GUN = Model(inputs=[generated_patch_input, raw_patch_input], outputs=[x], name='PatchGAN')
    #print('patchGAN summary')
    #Patch_GUN.summary()

    x = [Patch_GUN([list_input[i], list_raw_input[i]]) for i in range(patch_num)]

    if len(x)>1:
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

# Convolutional Auto Encoder
def CAE(img_shape, model_name='Conv_Auto_Encoder'):
    input_img = Input(shape=img_shape)
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)

    model = Model(input_img, decoded)

    return model

def load_generato(img_shape, disc_img_shape):
    model = generator_unet_upsampling(img_shape, disc_img_shape)
    model.summary()
    return model

def load_DCGAN_discriminator(img_shape, disc_img_shape, patch_num):
    model = DCGAN_discriminator(img_shape, disc_img_shape, patch_num)
    model.summary()
    return model

def load_DCGAN(generator, discriminator, img_shape, patch_size):
    model = DCGAN(generator, discriminator, img_shape, patch_size)
    return model

def load_CAE(img_shape):
    model = CAE(img_shape)
    model.summary()
    return model