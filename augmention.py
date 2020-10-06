# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:17:45 2020

@author: Administrator
"""

import os
import random
import glob
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf




filename =  r''
file_path =r''
file_data = glob.glob(file_path+'/*.npy')

label = np.array(pd.read_csv(filename))


for j in range(len(file_data)):
    img = np.load(file_data[j])
    for i in range(len(label)):
        if ((file_data[j].split('\\')[-1][:-4] == label[i][0]) and (label[i][1] == 0)):
            img1 = np.rot90(img,k=1,axes=(0,1))#表示在x轴旋转90*1度，k为正值代表逆时针旋转90度
            np.save(file_data[j][:-4] + '_1.npy', img1)

            # img2 = np.rot90(img,k=2,axes=(0,1))
            # np.save(file_data[j][:-4] + '_2.npy', img2)

            img3 = np.rot90(img,k=3,axes=(0,1))
            np.save(file_data[j][:-4] + '_3.npy', img3)

            # img4 = np.rot90(img,k=1,axes=(0,2))
            # np.save(file_data[j][:-4] + '_4.npy', img4)
            img5 = np.rot90(img,k=2,axes=(0,2))
            np.save(file_data[j][:-4] + '_5.npy', img5)
            # img6 = np.rot90(img,k=3,axes=(0,2))
            # np.save(file_data[j][:-4] + '_6.npy', img6)
            # img7 = np.rot90(img,k=1,axes=(1,2))
            # np.save(file_data[j][:-4] + '_7.npy', img7)
            # img8 = np.rot90(img,k=2,axes=(1,2))
            # np.save(file_data[j][:-4] + '_8.npy', img8)
            # img9 = np.rot90(img,k=3,axes=(1,2))
            # np.save(file_data[j][:-4] + '_9.npy', img9)

        elif ((file_data[j].split('\\')[-1][:-4] == label[i][0]) and (label[i][1] == 1)):
            img10 = np.rot90(img,k=1,axes=(0,1))
            np.save(file_data[j][:-4] + '_10.npy', img10)

            # img11 = np.rot90(img,k=2,axes=(0,1))
            # np.save(file_data[j][:-4] + '_11.npy', img11)

            img12 = np.rot90(img,k=3,axes=(0,1))
            np.save(file_data[j][:-4] + '_12.npy', img12)

            # img13 = np.rot90(img,k=1,axes=(0,2))
            # np.save(file_data[j][:-4] + '_13.npy', img13)

            img14 = np.rot90(img,k=2,axes=(0,2))
            np.save(file_data[j][:-4] + '_14.npy', img14)
            # img15 = np.rot90(img,k=3,axes=(0,2))
            # np.save(file_data[j][:-4] + '_15.npy', img15)
            img16 = np.rot90(img,k=1,axes=(1,2))
            np.save(file_data[j][:-4] + '_16.npy', img16)
            # img17 = np.rot90(img,k=2,axes=(1,2))
            # np.save(file_data[j][:-4] + '_17.npy', img17)
            # img18 = np.rot90(img,k=3,axes=(1,2))
            # np.save(file_data[j][:-4] + '_18.npy', img18)
            
            # img19 = np.flipud(img13)
            # np.save(file_data[j][:-4] + '_19.npy', img19)
            # img20 = np.flipud(img14)
            # np.save(file_data[j][:-4] + '_20.npy', img20)
            # img21 = np.flipud(img15)
            # np.save(file_data[j][:-4] + '_21.npy', img21)
            # img22 = np.flipud(img16)
            # np.save(file_data[j][:-4] + '_22.npy', img22)
            # img23 = np.flipud(img17)
            # np.save(file_data[j][:-4] + '_23.npy', img23)
            # img24 = np.flipud(img18)#这个是翻转操作，旋转以后再翻转
            # np.save(file_data[j][:-4] + '_24.npy', img24)
            

        elif ((file_data[j].split('\\')[-1][:-4] == label[i][0]) and (label[i][1] == 3)):
            img25 = np.rot90(img,k=1,axes=(0,1))
            np.save(file_data[j][:-4] + '_25.npy', img25)

            # img26 = np.rot90(img,k=2,axes=(0,1))
            # np.save(file_data[j][:-4] + '_26.npy', img26)

            img27 = np.rot90(img,k=3,axes=(0,1))
            np.save(file_data[j][:-4] + '_27.npy', img27)

            # img28 = np.rot90(img,k=1,axes=(0,2))
            # np.save(file_data[j][:-4] + '_28.npy', img28)
            img29 = np.rot90(img,k=2,axes=(0,2))
            np.save(file_data[j][:-4] + '_29.npy', img29)
            # img30 = np.rot90(img,k=3,axes=(0,2))
            # np.save(file_data[j][:-4] + '_30.npy', img30)
            # img31 = np.rot90(img,k=1,axes=(1,2))
            # np.save(file_data[j][:-4] + '_31.npy', img31)
            # img32 = np.rot90(img,k=2,axes=(1,2))
            # np.save(file_data[j][:-4] + '_32.npy', img32)
            # img33 = np.rot90(img,k=3,axes=(1,2))
            # np.save(file_data[j][:-4] + '_33.npy', img33)

        else:
            a = 1

            


print('finish')


a = [88,46,765,77]
b = np.sum(a)/a
b = b/np.sum(b)
print(b)