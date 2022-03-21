# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:49:09 2021

@author: Mutiara Saviera
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

from sklearn.model_selection import train_test_split


list_images=glob.glob('D:/Mutiara Saviera/UNSRI/TUGAS AKHIR/Referensi/Optic Disk/Dataset/Base11/*.tif')
images_list=[]

for i in range(len(list_images)):
  img=cv2.imread(list_images[i])        
  img=cv2.resize(img, (512,512))
  images_list.append(img)  
  
images_list = np.asarray(images_list)
plt.imshow(images_list[0], cmap="bone")
images_rgb=[]

for i in range(images_list.shape[0]):
  im_rgb = cv2.cvtColor(images_list[i], cv2.COLOR_BGR2RGB)
  images_rgb.append(im_rgb)  

images_rgb = np.asarray(images_rgb)
plt.imshow(images_rgb[0], cmap="bone")

green_image = images_rgb[:,:,:,1]
green_image = np.asarray(green_image)

plt.imshow(green_image[0], cmap="bone")


file_name = [os.path.basename(x) for x in list_images]
dict = {'nama_file': file_name, 'image': images_list}
dict

gt=glob.glob('D:/Mutiara Saviera/UNSRI/TUGAS AKHIR/Referensi/Optic Disk/Dataset/Base11/gtfinal/*.png')
gt_list=[]
for i in range(len(gt)):
    img_gt=cv2.imread(gt[i])
    img_gt=cv2.resize(img_gt, (512,512))
    gt_list.append(img_gt)    

gt_list = np.asarray(gt_list)
gt_list.shape
 
X_train, X_test, y_train, y_test = train_test_split(green_image, gt_list, test_size=0.2, random_state=1)

np.save('x_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', X_test)
np.save('y_test.npy', y_test)


