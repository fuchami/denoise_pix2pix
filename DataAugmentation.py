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
    images = glob.glob(args.savepath + 'noise_train/' + '*.png')

    for i in images:
        print(i)
        src = cv2.imread(i, 1)

        # 反転
        hflip_img = cv2.flip(src, 1)
        # 上下反転
        vflip_img = cv2.flip(src, 0)

        root, ext = os.path.splitext(i)
        cv2.imwrite(root + '_hflip.png', hflip_img)
        cv2.imwrite(root + '_vflip.png', vflip_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Augmentation script')
    parser.add_argument('--datapath' , '-p', type=str, require=True)

    args = parser.parse_args()
    
    main(args)