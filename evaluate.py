# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:49:09 2021

@author: Mutiara Saviera
"""
import os
import cv2
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow
from extract_patch import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from evaluate_function import *
from model import *
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
from sklearn.metrics import classification_report

x_test=np.load("x_test.npy")
x_test=x_test.astype(np.float32)
y_test=np.load("y_test.npy")
y_test=y_test.astype(np.float32)
#x_test=np.expand_dims(x_test,-1)

x_test/=255
y_test/=255

x_test_patch = paint_border_overlap(x_test, 64, 64, 32, 32)
y_test_patch = paint_border_overlap(y_test, 64, 64, 32, 32)

x_test_patch=extract_ordered_overlap(x_test_patch,64,64,32,32)
print (x_test_patch.shape)

#for i in range(100) :
    #aa=f"D:/optic disk/patchimage/"
    #bb=".png"
    #ff=str(i)
    #plt.imsave(aa+ff+bb, x_test_patch[i])

model=dsnet(input_size=(64,64,3))
model.load_weights('newdsnet100.h5')
predicx=model.predict(x_test_patch)

y_pred = recompone_overlap(predicx, 512, 512, 32, 32)

#np.save('y_pred.npy', y_pred)

y_pred*=255

y_pred=y_pred.astype(np.uint8)

for i in range(y_pred.shape[0]):
    pred=cv2.imwrite(os.path.join('D:/optic disk', 'image_'+str(i)+'.png'), y_pred[i])

    
y_pred=np.argmax(y_pred,axis=-1)
y_pred=np.expand_dims(y_pred,axis=-1)

y_test=np.argmax(y_test,axis=-1)
y_test=np.expand_dims(y_test,axis=-1)

y_pred=y_pred.flatten()
y_test=y_test.flatten()
    
plot_cm(y_test, y_pred)

f1score = f1_score(y_test, y_pred, average='macro')
acc = accuracy_score(y_test, y_pred)
iouoc = iou_oc(y_test, y_pred)
iouod = iou_od(y_test, y_pred)
meaniou = mean_iou(y_test, y_pred)

print('Accuracy:', acc)
#print('IoU OD (Jaccard Index):', iouod)
#print('IoU OD (Jaccard Index):', iouoc)
#print('Mean IoU (Jaccard Index):', meaniou)


print(classification_report(y_test, y_pred))

print(multilabel_confusion_matrix(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))



