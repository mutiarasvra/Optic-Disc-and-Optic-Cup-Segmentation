# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:30:09 2021

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
import seaborn as sns

def unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(3, (1, 1), activation='softmax')(conv9)
    return Model(inputs=[inputs], outputs=[conv10])

def unet_residual(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
     
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    # Residual Block
    x = Conv2D(512, (3, 3), padding="same")(conv5)
    y = BatchNormalization()(x)
    z = Activation("relu")(y)
   
    a = Conv2D(512, (3, 3), padding="same")(z)

    s = Add()([conv5, a])
    b = BatchNormalization()(s)
    z = Activation("relu")(b)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(z), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv5 = BatchNormalization()(conv5)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(3, (1, 1), activation='softmax')(conv9)
    return Model(inputs=[inputs], outputs=[conv10])

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def segnet(input_size=(256,256,3)):
    inputs = Input(input_size)
 
    #encoder
    #block1
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv2 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    #block2
    conv3 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    conv4 = Conv2D(64, (3, 3), padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    #block3
    conv5 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    conv6 = Conv2D(128, (3, 3), padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)
    conv7 = Conv2D(128, (3, 3), padding='same')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)
    
    #block4
    conv8 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)
    conv9 = Conv2D(256, (3, 3), padding='same')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    conv10 = Conv2D(256, (3, 3), padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation("relu")(conv10)    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv10)
    
    #block5
    conv11 = Conv2D(512, (3, 3), padding='same')(pool4)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation("relu")(conv11)
    conv12 = Conv2D(512, (3, 3), padding='same')(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation("relu")(conv12)
    conv13 = Conv2D(512, (3, 3), padding='same')(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation("relu")(conv13)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv13)
    
    #decoder
    up1 = UpSampling2D(size=(2,2))(pool5)
    conv14 = Conv2D(512, (3, 3), padding='same')(up1)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation("relu")(conv14)
    conv15 = Conv2D(512, (3, 3), padding='same')(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation("relu")(conv15)
    conv16 = Conv2D(512, (3, 3), padding='same')(conv15)
    conv16 = BatchNormalization()(conv16)
    conv16 = Activation("relu")(conv16)

    up2 = UpSampling2D(size=(2,2))(conv16)
    conv17 = Conv2D(256, (3, 3), padding='same')(up2)
    conv17 = BatchNormalization()(conv17)
    conv17 = Activation("relu")(conv17)
    conv18 = Conv2D(256, (3, 3), padding='same')(conv17)
    conv18 = BatchNormalization()(conv18)
    conv18 = Activation("relu")(conv18)
    conv19 = Conv2D(256, (3, 3), padding='same')(conv18)
    conv19 = BatchNormalization()(conv19)
    conv19 = Activation("relu")(conv19)
    
    up3 = UpSampling2D(size=(2,2))(conv19)
    conv20 = Conv2D(128, (3, 3), padding='same')(up3)
    conv20 = BatchNormalization()(conv20)
    conv20 = Activation("relu")(conv20)
    conv21 = Conv2D(128, (3, 3), padding='same')(conv20)
    conv21 = BatchNormalization()(conv21)
    conv21 = Activation("relu")(conv21)
    conv22 = Conv2D(128, (3, 3), padding='same')(conv21)
    conv22 = BatchNormalization()(conv22)
    conv22 = Activation("relu")(conv22)
    
    up4 = UpSampling2D(size=(2,2))(conv22)
    conv23 = Conv2D(64, (3, 3), padding='same')(up4)
    conv23 = BatchNormalization()(conv23)
    conv23 = Activation("relu")(conv23)
    conv24 = Conv2D(64, (3, 3), padding='same')(conv23)
    conv24 = BatchNormalization()(conv24)
    conv24 = Activation("relu")(conv24)
    
    up5 = UpSampling2D(size=(2,2))(conv24)
    conv25 = Conv2D(32, (3, 3), padding='same')(up5)
    conv25 = BatchNormalization()(conv25)
    conv25 = Activation("relu")(conv25)
    conv26 = Conv2D(32, (3, 3), padding='same')(conv25)
    conv26 = BatchNormalization()(conv26)
    conv26 = Activation("relu")(conv26)
    
    conv27 = Conv2D(3, (1, 1), activation='softmax')(conv26)
        
    return Model(inputs=[inputs], outputs=[conv27])

def dsnet(input_size=(256,256,3)):
    inputs = Input(input_size)
 
    #encoder
    #block1
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv2 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    #block2
    conv3 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    conv4 = Conv2D(64, (3, 3), padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    #block3
    conv5 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    conv6 = Conv2D(128, (3, 3), padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)
    conv7 = Conv2D(128, (3, 3), padding='same')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)
    
    #block4
    conv8 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)
    conv9 = Conv2D(256, (3, 3), padding='same')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    conv10 = Conv2D(256, (3, 3), padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation("relu")(conv10)    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv10)
    
    #block5
    conv11 = Conv2D(512, (3, 3), padding='same')(pool4)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation("relu")(conv11)
    conv12 = Conv2D(512, (3, 3), padding='same')(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation("relu")(conv12)
    conv13 = Conv2D(512, (3, 3), padding='same')(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation("relu")(conv13)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv13)
    
    # D1
    con1 = BatchNormalization()(pool5)
    con1 = Activation("relu")(con1)
    con1 = Conv2D(512, (1, 1), padding='same')(con1)
    con2 = BatchNormalization()(con1)
    con2 = Activation("relu")(con2)
    con2 = Conv2D(512, (3, 3), padding='same')(con2)
    merge_dense = concatenate([con2,pool5], axis = 3)
    
    # D2
    con3 = BatchNormalization()(merge_dense)
    con3 = Activation("relu")(con3)
    con3 = Conv2D(512, (1, 1), padding='same')(con3)
    con4 = BatchNormalization()(con3)
    con4 = Activation("relu")(con4)
    con4 = Conv2D(512, (3, 3), padding='same')(con4)
    merge_dense1 = concatenate([con4,merge_dense], axis = 3)
    
    # D3
    con5 = BatchNormalization()(merge_dense1)
    con5 = Activation("relu")(con5)
    con5 = Conv2D(512, (1, 1), padding='same')(con5)
    con6 = BatchNormalization()(con5)
    con6 = Activation("relu")(con6)
    con6 = Conv2D(512, (3, 3), padding='same')(con6)
    merge_dense2 = concatenate([con6,merge_dense1], axis = 3)
    
    #decoder
    up1 = UpSampling2D(size=(2,2))(merge_dense2)
    conv14 = Conv2D(512, (3, 3), padding='same')(up1)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation("relu")(conv14)
    conv15 = Conv2D(512, (3, 3), padding='same')(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation("relu")(conv15)
    conv16 = Conv2D(512, (3, 3), padding='same')(conv15)
    conv16 = BatchNormalization()(conv16)
    conv16 = Activation("relu")(conv16)

    up2 = UpSampling2D(size=(2,2))(conv16)
    conv17 = Conv2D(256, (3, 3), padding='same')(up2)
    conv17 = BatchNormalization()(conv17)
    conv17 = Activation("relu")(conv17)
    conv18 = Conv2D(256, (3, 3), padding='same')(conv17)
    conv18 = BatchNormalization()(conv18)
    conv18 = Activation("relu")(conv18)
    conv19 = Conv2D(256, (3, 3), padding='same')(conv18)
    conv19 = BatchNormalization()(conv19)
    conv19 = Activation("relu")(conv19)
    
    up3 = UpSampling2D(size=(2,2))(conv19)
    conv20 = Conv2D(128, (3, 3), padding='same')(up3)
    conv20 = BatchNormalization()(conv20)
    conv20 = Activation("relu")(conv20)
    conv21 = Conv2D(128, (3, 3), padding='same')(conv20)
    conv21 = BatchNormalization()(conv21)
    conv21 = Activation("relu")(conv21)
    conv22 = Conv2D(128, (3, 3), padding='same')(conv21)
    conv22 = BatchNormalization()(conv22)
    conv22 = Activation("relu")(conv22)
    
    up4 = UpSampling2D(size=(2,2))(conv22)
    conv23 = Conv2D(64, (3, 3), padding='same')(up4)
    conv23 = BatchNormalization()(conv23)
    conv23 = Activation("relu")(conv23)
    conv24 = Conv2D(64, (3, 3), padding='same')(conv23)
    conv24 = BatchNormalization()(conv24)
    conv24 = Activation("relu")(conv24)
    
    up5 = UpSampling2D(size=(2,2))(conv24)
    conv25 = Conv2D(32, (3, 3), padding='same')(up5)
    conv25 = BatchNormalization()(conv25)
    conv25 = Activation("relu")(conv25)
    conv26 = Conv2D(32, (3, 3), padding='same')(conv25)
    conv26 = BatchNormalization()(conv26)
    conv26 = Activation("relu")(conv26)
    
    conv27 = Conv2D(3, (1, 1), activation='softmax')(conv26)
        
    return Model(inputs=[inputs], outputs=[conv27])

def dsnet1(input_size=(256,256,3)):
    inputs = Input(input_size)
 
    #encoder
    #block1
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv2 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    #block2
    conv3 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    conv4 = Conv2D(64, (3, 3), padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    #block3
    conv5 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    conv6 = Conv2D(128, (3, 3), padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)
    conv7 = Conv2D(128, (3, 3), padding='same')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)
    
    #block4
    conv8 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)
    conv9 = Conv2D(256, (3, 3), padding='same')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    conv10 = Conv2D(256, (3, 3), padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation("relu")(conv10)    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv10)
    
    # D1
    con1 = BatchNormalization()(pool4)
    con1 = Activation("relu")(con1)
    con1 = Conv2D(512, (1, 1), padding='same')(con1)
    con2 = BatchNormalization()(con1)
    con2 = Activation("relu")(con2)
    con2 = Conv2D(512, (3, 3), padding='same')(con2)
    merge_dense = concatenate([con2,pool4], axis = 3)
    
    # D2
    con3 = BatchNormalization()(merge_dense)
    con3 = Activation("relu")(con3)
    con3 = Conv2D(512, (1, 1), padding='same')(con3)
    con4 = BatchNormalization()(con3)
    con4 = Activation("relu")(con4)
    con4 = Conv2D(512, (3, 3), padding='same')(con4)
    merge_dense1 = concatenate([con4,merge_dense], axis = 3)
    
    # D3
    con5 = BatchNormalization()(merge_dense1)
    con5 = Activation("relu")(con5)
    con5 = Conv2D(512, (1, 1), padding='same')(con5)
    con6 = BatchNormalization()(con5)
    con6 = Activation("relu")(con6)
    con6 = Conv2D(512, (3, 3), padding='same')(con6)
    merge_dense2 = concatenate([con6,merge_dense1], axis = 3)
    
    #decoder
    up1 = UpSampling2D(size=(2,2))(merge_dense2)
    conv14 = Conv2D(512, (3, 3), padding='same')(up1)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation("relu")(conv14)
    conv15 = Conv2D(512, (3, 3), padding='same')(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation("relu")(conv15)
    conv16 = Conv2D(512, (3, 3), padding='same')(conv15)
    conv16 = BatchNormalization()(conv16)
    conv16 = Activation("relu")(conv16)
    
    up3 = UpSampling2D(size=(2,2))(conv16)
    conv20 = Conv2D(128, (3, 3), padding='same')(up3)
    conv20 = BatchNormalization()(conv20)
    conv20 = Activation("relu")(conv20)
    conv21 = Conv2D(128, (3, 3), padding='same')(conv20)
    conv21 = BatchNormalization()(conv21)
    conv21 = Activation("relu")(conv21)
    conv22 = Conv2D(128, (3, 3), padding='same')(conv21)
    conv22 = BatchNormalization()(conv22)
    conv22 = Activation("relu")(conv22)
    
    up4 = UpSampling2D(size=(2,2))(conv22)
    conv23 = Conv2D(64, (3, 3), padding='same')(up4)
    conv23 = BatchNormalization()(conv23)
    conv23 = Activation("relu")(conv23)
    conv24 = Conv2D(64, (3, 3), padding='same')(conv23)
    conv24 = BatchNormalization()(conv24)
    conv24 = Activation("relu")(conv24)
    
    up5 = UpSampling2D(size=(2,2))(conv24)
    conv25 = Conv2D(32, (3, 3), padding='same')(up5)
    conv25 = BatchNormalization()(conv25)
    conv25 = Activation("relu")(conv25)
    conv26 = Conv2D(32, (3, 3), padding='same')(conv25)
    conv26 = BatchNormalization()(conv26)
    conv26 = Activation("relu")(conv26)
    
    conv27 = Conv2D(3, (1, 1), activation='softmax')(conv26)
        
    return Model(inputs=[inputs], outputs=[conv27])


def dsnet2(input_size=(256,256,3)):
    inputs = Input(input_size)
 
    #encoder
    #block1
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv2 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    #block2
    conv3 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    conv4 = Conv2D(64, (3, 3), padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    #block3
    conv5 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    conv6 = Conv2D(128, (3, 3), padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)
    conv7 = Conv2D(128, (3, 3), padding='same')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)
    
    #block4
    conv8 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)
    conv9 = Conv2D(256, (3, 3), padding='same')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    conv10 = Conv2D(256, (3, 3), padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation("relu")(conv10)    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv10)
    
    #block5
    conv11 = Conv2D(512, (3, 3), padding='same')(pool4)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation("relu")(conv11)
    conv12 = Conv2D(512, (3, 3), padding='same')(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation("relu")(conv12)
    conv13 = Conv2D(512, (3, 3), padding='same')(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation("relu")(conv13)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv13)
    
    # D1
    con1 = BatchNormalization()(pool5)
    con1 = Activation("relu")(con1)
    con1 = Conv2D(512, (1, 1), padding='same')(con1)
    con2 = BatchNormalization()(con1)
    con2 = Activation("relu")(con2)
    con2 = Conv2D(512, (3, 3), padding='same')(con2)
    merge_dense = concatenate([con2,pool5], axis = 3)
    
    # D2
    con3 = BatchNormalization()(merge_dense)
    con3 = Activation("relu")(con3)
    con3 = Conv2D(512, (1, 1), padding='same')(con3)
    con4 = BatchNormalization()(con3)
    con4 = Activation("relu")(con4)
    con4 = Conv2D(512, (3, 3), padding='same')(con4)
    merge_dense1 = concatenate([con4,merge_dense], axis = 3)
    
    
    #decoder
    up1 = UpSampling2D(size=(2,2))(merge_dense1)
    conv14 = Conv2D(512, (3, 3), padding='same')(up1)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation("relu")(conv14)
    conv15 = Conv2D(512, (3, 3), padding='same')(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation("relu")(conv15)
    conv16 = Conv2D(512, (3, 3), padding='same')(conv15)
    conv16 = BatchNormalization()(conv16)
    conv16 = Activation("relu")(conv16)

    up2 = UpSampling2D(size=(2,2))(conv16)
    conv17 = Conv2D(256, (3, 3), padding='same')(up2)
    conv17 = BatchNormalization()(conv17)
    conv17 = Activation("relu")(conv17)
    conv18 = Conv2D(256, (3, 3), padding='same')(conv17)
    conv18 = BatchNormalization()(conv18)
    conv18 = Activation("relu")(conv18)
    conv19 = Conv2D(256, (3, 3), padding='same')(conv18)
    conv19 = BatchNormalization()(conv19)
    conv19 = Activation("relu")(conv19)
    
    up3 = UpSampling2D(size=(2,2))(conv19)
    conv20 = Conv2D(128, (3, 3), padding='same')(up3)
    conv20 = BatchNormalization()(conv20)
    conv20 = Activation("relu")(conv20)
    conv21 = Conv2D(128, (3, 3), padding='same')(conv20)
    conv21 = BatchNormalization()(conv21)
    conv21 = Activation("relu")(conv21)
    conv22 = Conv2D(128, (3, 3), padding='same')(conv21)
    conv22 = BatchNormalization()(conv22)
    conv22 = Activation("relu")(conv22)
    
    up4 = UpSampling2D(size=(2,2))(conv22)
    conv23 = Conv2D(64, (3, 3), padding='same')(up4)
    conv23 = BatchNormalization()(conv23)
    conv23 = Activation("relu")(conv23)
    conv24 = Conv2D(64, (3, 3), padding='same')(conv23)
    conv24 = BatchNormalization()(conv24)
    conv24 = Activation("relu")(conv24)
    
    up5 = UpSampling2D(size=(2,2))(conv24)
    conv25 = Conv2D(32, (3, 3), padding='same')(up5)
    conv25 = BatchNormalization()(conv25)
    conv25 = Activation("relu")(conv25)
    conv26 = Conv2D(32, (3, 3), padding='same')(conv25)
    conv26 = BatchNormalization()(conv26)
    conv26 = Activation("relu")(conv26)
    
    conv27 = Conv2D(3, (1, 1), activation='softmax')(conv26)
        
    return Model(inputs=[inputs], outputs=[conv27])

def FCN8(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    #encoder
    #block1
    con1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    con1 = Conv2D(32, (3, 3), activation='relu', padding='same')(con1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(con1)
    
    #block2
    con2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    con2 = Conv2D(64, (3, 3), activation='relu', padding='same')(con2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(con2)
    
    #block3
    con3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    con3 = Conv2D(128, (3, 3), activation='relu', padding='same')(con3)
    con3 = Conv2D(128, (3, 3), activation='relu', padding='same')(con3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(con3)
    
    #block4
    con4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    con4 = Conv2D(256, (3, 3), activation='relu', padding='same')(con4)
    con4 = Conv2D(256, (3, 3), activation='relu', padding='same')(con4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(con4)## (None, 14, 14, 512)
    
    #block5
    con5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    con5 = Conv2D(512, (3, 3), activation='relu', padding='same')(con5)
    con5 = Conv2D(512, (3, 3), activation='relu', padding='same')(con5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(con5)## (None, 7, 7, 512)

    con6 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool5)  
    con7 = Conv2D(512, (3, 3), activation='relu', padding='same')(con6)
    
    ## 4 times upsamping for pool4 layer
    con7_4 = Conv2DTranspose(512, kernel_size=(4,4),  strides=(4,4))(con7)
    
    ## 2 times upsampling for pool411
    pool411_2 = Conv2DTranspose(512 , kernel_size=(2,2),  strides=(2,2))(pool4)
    
    pool311 = Conv2D(512, (1 , 1) , activation='relu' , padding='same', name="pool3_11")(pool3)
        
    o = Add(name="add")([pool411_2, pool311, con7_4 ])
    o = Conv2DTranspose(512, kernel_size=(8,8) ,  strides=(8,8))(o)
    
    
    o = Conv2D(3, 1, activation = 'softmax')(o)
    
    return Model(inputs=[inputs], outputs=[o])

def UnetLSTM (input_size = (256,256,1)):
    N = input_size[0]
    inputs = Input(input_size) 
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)

    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)
    merge6  = concatenate([x1,x2], axis = 1) 
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(conv2)
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)
    merge7  = concatenate([x1,x2], axis = 1) 
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
        
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)    

    x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)
    merge8  = concatenate([x1,x2], axis = 1) 
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)    
    
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv9 = Conv2D(3, 1, activation = 'softmax')(conv8)
    conv9 = Conv2D(3, 1, activation = 'softmax')(conv8)

    model = Model(inputs = inputs , outputs = conv9)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def fcnu8(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    #encoder
    #block1
    con1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    con1 = Conv2D(32, (3, 3), activation='relu', padding='same')(con1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(con1)
   
    #block2
    con2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    con2 = Conv2D(64, (3, 3), activation='relu', padding='same')(con2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(con2)
       
    
    #block3
    con3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    con3 = Conv2D(128, (3, 3), activation='relu', padding='same')(con3)
    con3 = Conv2D(128, (3, 3), activation='relu', padding='same')(con3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(con3)
   
    
    #block4
    con4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    con4 = Conv2D(256, (3, 3), activation='relu', padding='same')(con4)
    con4 = Conv2D(256, (3, 3), activation='relu', padding='same')(con4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(con4)## (None, 14, 14, 512)
       
    
    #block5
    con5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    con5 = Conv2D(512, (3, 3), activation='relu', padding='same')(con5)
    con5 = Conv2D(512, (3, 3), activation='relu', padding='same')(con5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(con5)## (None, 7, 7, 512)
       

    con6 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool5)  
    con7 = Conv2D(512, (3, 3), activation='relu', padding='same')(con6)
    
    ## 4 times upsamping for pool4 layer
    con7_4 = Conv2DTranspose(512, kernel_size=(4,4),  strides=(4,4))(con7)
    
    ## 2 times upsampling for pool411
    pool411_2 = Conv2DTranspose(512 , kernel_size=(2,2),  strides=(2,2))(pool4)
    
    pool311 = Conv2D(512, (1 , 1) , activation='relu' , padding='same', name="pool3_11")(pool3)
        
    o = Add(name="add")([pool411_2, pool311, con7_4 ])
    o = Conv2DTranspose(512, kernel_size=(8,8) ,  strides=(8,8))(o)
    
    #decoder
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(con5), con4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), con3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), con2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), con1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(3, (1, 1), activation='softmax')(conv9)
    return Model(inputs=[inputs], outputs=[conv10])
    
