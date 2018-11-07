# coding:utf-8
"""
Data Augmentation
左右反転・上下反転したものを取得

"""

import os,sys
import glob
import argparse
import numpy as np
from PIL import Image
import cv2

def main(args):

    """ source datasets path load """
    images = glob.glob(args.datapath + 'noise_train/' + '*.png')

    for i in images:
        print(i)
        basename = os.path.basename(i)
        print (basename)

        """ load noise images """
        noise_src = cv2.imread(args.datapath + 'noise_train/' + basename, 1)

        # 反転
        hflip_img = cv2.flip(noise_src, 1)
        # 上下反転
        vflip_img = cv2.flip(noise_src, 0)

        # save
        cv2.imwrite(args.datapath+ 'noise_train/hflip_' + basename, hflip_img)
        cv2.imwrite(args.datapath+ 'noise_train/vflip_' + basename, vflip_img)

        """ load truth images """
        truth_src = cv2.imread(args.datapath + 'truth_train/' + basename, 1)

        # 反転
        hflip_img = cv2.flip(noise_src, 1)
        # 上下反転
        vflip_img = cv2.flip(noise_src, 0)

        # save
        cv2.imwrite(args.datapath+ 'truth_train/hflip_' + basename, hflip_img)
        cv2.imwrite(args.datapath+ 'truth_train/vflip_' + basename, vflip_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Augmentation script')
    parser.add_argument('--datapath' , '-p', type=str, required=True)

    args = parser.parse_args()
    
    main(args)