# coding:utf-8
"""
ノイズを乗っけるスクリプト

"""

import os,sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob

def main():

    # original path
    origin_path = '/Users/yuuki/Desktop/python_code/original_datasets/cifar/train/'
    save2_path = '/Users/yuuki/Desktop/python_code/datasets/'
    img_path_list = glob.glob(origin_path+'/*.png')
    #print(img_path_list)

    cnt = 0

    for img_path in img_path_list:
        print(img_path)

        img = cv2.imread(img_path)
        # そのまま保存
        cv2.imwrite(save2_path+'truth_train/'+str(cnt)+'.png', img)

        row,col,ch = img.shape
        # ごましお比率
        s_vs_p = 0.5
        amount = 0.004
        sp_img = img.copy()
        
        num_salt = np.ceil(amount* img.size * s_vs_p)
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape]
        sp_img[coords[:-1]] = (255,255,255)
        
        num_papper = np.ceil(amount* img.size * (1. -s_vs_p))
        coords = [np.random.randint(0, i-1, int(num_papper)) for i in img.shape]
        sp_img[coords[:-1]] = (0,0,0)

        cv2.imwrite(save2_path+'noise_train/'+str(cnt)+'.png', sp_img)

        cnt += 1

if __name__ == '__main__':

    main()
