# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:49:09 2021

@author: Mutiara Saviera
"""
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras.layers import *
import numpy as np
import tensorflow as tf
from extract_patch import extract_random
from model import *
from keras_unet.utils import plot_segm_history

x_train=np.load("x_train.npy")
x_train=x_train.astype(np.float32)
y_train=np.load("y_train.npy")
y_train=y_train.astype(np.float32)
#x_train=np.expand_dims(x_train,-1)

x_train/=255
y_train/=255

x_train_patch,y_train_patch=extract_random(x_train,y_train,64,64,16000)

#del x_train,y_train

input_shape = x_train_patch.shape[1:]
model= fcnu8(input_size=(64,64,3))

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy',
                  metrics=[dice_coef, 'categorical_accuracy'])
model.summary()
ms=ModelCheckpoint("newfcnu.h5", monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch')

history = model.fit(x_train_patch,y_train_patch,epochs=200,batch_size=8,validation_split=0.3,shuffle=True,callbacks=[ms])

plot_segm_history(
    history, # required - keras training history object
    metrics=['categorical_accuracy', 'val_categorical_accuracy'], # optional - metrics names to plot
    losses=['loss', 'val_loss']) # optional - loss names to plot

