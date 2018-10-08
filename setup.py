# coding:utf-8

"""
データを揃える前処理スクリプト

"""

import os, sys
import glob
import random
import shutil

def main(src_dir, tar_dir, noise_path, truth_path):

    # パス内の画像リストの取得
    img_list = os.listdir(src_dir + noise_path + '*.png')
    # 取得した画像をランダムにシャッフル
    random.shuffle(img_list)

    if not os.path.exists(tar_dir+'noise_train/'):
        os.makedirs(tar_dir+'nose_train')
    if not os.path.exists(tar_dir+'noise_val'):
        os.makedirs(tar_dir+'noise_val')
    if not os.path.exists(tar_dir+'truth_train'):
        os.makedirs(tar_dir+'truth_train')
    if not os.path.exists(tar_dir+'truth_val'):
        os.makedirs(tar_dir+'truth_val')
    

    # 入力画像
    for i in range(len(img_list)):
        shutil.copyfile("%s/%s/%s" % (src_dir, noise_path, img_list[i]),
                            "%s/%s/img%04d.png" % (tar_dir, 'noise_train', i))

    img_list = os.listdir(tar_dir + noise_path )
    random.shuffle(img_list)
    TEST_SIZE = int(len(img_list)/10)
    # 検証データに分割
    for i in range (TEST_SIZE):
        shutil.rename("%s/%s/%s" % (tar_dir, noise_path, img_list[i]),
                            "%s/%s/img%04d.png" % (tar_dir, 'noise_val', i))

    # 出力画像
    for i in range(len(img_list)):
        shutil.copyfile("%s/%s/%s" % (src_dir, truth_path, img_list[i]),
                            "%s/%s/img%04d.png" % (tar_dir, 'truth_train', i))

    img_list = os.listdir(tar_dir + truth_path )
    random.shuffle(img_list)
    TEST_SIZE = int(len(img_list)/10)
    # 検証データに分割
    for i in range (TEST_SIZE):
        shutil.rename("%s/%s/%s" % (tar_dir, truth_path, img_list[i]),
                            "%s/%s/img%04d.png" % (tar_dir, 'truth_val', i))


if __name__ == '__main__' :
    source_dir = '/media/futami/HDD1/DATASET_KINGDOM/ORIGINAL_DATA/'
    target_dir = '/media/futami/HDD1/DATASET_KINGDOM/denoise/'

    noise_path = '0915before/'
    truth_path = '0915after/'

    main(source_dir, target_dir, noise_path, truth_path)