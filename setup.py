# coding:utf-8

"""
データを揃える前処理スクリプト
1:9で訓練データと検証データに分割する

"""

import os, sys
import glob
import random
import shutil
import argparse

def main(args):

    if not os.path.exists(args.targetdir+'noise_train/'):
        os.makedirs(args.targetdir+'noise_train')
    if not os.path.exists(args.targetdir+'noise_val'):
        os.makedirs(args.targetdir+'noise_val')
    if not os.path.exists(args.targetdir+'truth_train'):
        os.makedirs(args.targetdir+'truth_train')
    if not os.path.exists(args.targetdir+'truth_val'):
        os.makedirs(args.targetdir+'truth_val')
    
    # パス内の画像リストの取得
    img_list = os.listdir(args.sourcedir + args.noisepath)
    # 取得した画像をランダムにシャッフル
    random.shuffle(img_list)

    # 移動
    for i in range(len(img_list)):
        shutil.copyfile("%s%s%s" % (args.sourcedir, args.noisepath, img_list[i]),
                            "%s%s/img%04d.png" % (args.targetdir, 'noise_train', i))
        
        img_list[i] = img_list[i].replace('before', 'after')
        shutil.copyfile("%s%s%s" % (args.sourcedir, args.truthpath, img_list[i]),
                            "%s%s/img%04d.png" % (args.targetdir, 'truth_train', i))

    img_list = os.listdir(args.targetdir + 'noise_train')
    random.shuffle(img_list)

    TEST_SIZE = int(len(img_list)/10)
    for i in range (TEST_SIZE):
        os.rename("%s%s/%s" % (args.targetdir, 'noise_train', img_list[i]),
                            "%s%s/img%04d.png" % (args.targetdir, 'noise_val', i))

        img_list[i] = img_list[i].replace('before', 'after')
        os.rename("%s%s/%s" % (args.targetdir, 'truth_train', img_list[i]),
                            "%s%s/img%04d.png" % (args.targetdir, 'truth_val', i))
    

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='setup image data')
    parser.add_argument('--sourcedir', '-s', type=str, required=True)
    parser.add_argument('--targetdir', '-t', type=str, required=True)
    parser.add_argument('--noisepath', '-np', type=str, required=True)
    parser.add_argument('--truthpath', '-tp', type=str, required=True)

    args = parser.parse_args()

    main(args)