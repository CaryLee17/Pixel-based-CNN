import os
from PIL import Image
import numpy as np
from keras.layers import *
from keras.models import *
from keras import optimizers
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt
import tifffile as tif
def img_lab_read():
    path = r'D:\研究生\影像解译\色楞'
    lab = np.zeros((43,512,512))
    img = np.zeros((43,512,512,7))
    lab_name = os.listdir(os.path.join(path,'label'))
    for i,n in zip(range(len(lab_name)),lab_name):
        lab[i] = np.array(Image.open(path+'\\label\\'+n))
        img[i] = tif.imread(path+'\\crop\\'+n.split('.')[0]+'.tif').transpose([1,2,0])/2**16
    return lab, img
def one_hot(lab, labels_num = 2):
    num = lab.shape[0]
    x = lab.shape[1]
    y = lab.shape[2]
    one_hot_lab = np.zeros((num, x, y, labels_num))#比已知类比多一个背景
    for f in range(num):
        for i in range(labels_num):
            one_hot_lab[f,:,:,i] = (lab[f] == i+1)*1
    return one_hot_lab
lab, img = img_lab_read()
inputs = Input((512, 512,7))
width = 8
conv1 = Conv2D(width, (3, 3), activation='relu', padding='same')(inputs)
conv1 = BatchNormalization(trainable=False)(conv1)
conv1 = Conv2D(width, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(width*2, (3, 3), activation='relu', padding='same')(pool1)
conv2 = BatchNormalization(trainable=False)(conv2)
conv2 = Conv2D(width*2, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(width*4, (3, 3), activation='relu', padding='same')(pool2)
conv3 = BatchNormalization(trainable=False)(conv3)
conv3 = Conv2D(width*4, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(width*8, (3, 3), activation='relu', padding='same')(pool3)
conv4 = BatchNormalization(trainable=False)(conv4)
conv4 = Conv2D(width*8, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(width*16, 3, activation='relu', padding='same')(pool4)
conv5 = BatchNormalization(trainable=False)(conv5)
conv5 = Conv2D(width*16, 3, activation='relu', padding='same')(conv5)

up6 = UpSampling2D(size=(2, 2))(conv5)
up6 = concatenate([up6, conv4])
conv6 = SpatialDropout2D(0.35)(up6)
conv6 = Conv2D(width*8, (3, 3), activation='relu', padding='same')(conv6)
conv6 = Conv2D(width*8, (3, 3), activation='relu', padding='same')(conv6)

up7 = UpSampling2D(size=(2, 2))(conv6)
up7 = concatenate([up7, conv3])
conv7 = SpatialDropout2D(0.35)(up7)
conv7 = Conv2D(width*4, (3, 3), activation='relu', padding='same')(conv7)
conv7 = Conv2D(width*4, (3, 3), activation='relu', padding='same')(conv7)

up8 = UpSampling2D(size=(2, 2))(conv7)
up8 = concatenate([up8, conv2])
conv8 = SpatialDropout2D(0.35)(up8)
conv8 = Conv2D(width*2, (3, 3), activation='relu', padding='same')(conv8)
conv8 = Conv2D(width*2, (3, 3), activation='relu', padding='same')(conv8)

up9 = UpSampling2D(size=(2, 2))(conv8)
up9 = concatenate([up9, conv1])
conv9 = SpatialDropout2D(0.35)(up9)
conv9 = Conv2D(width, (3, 3), activation='relu',padding = 'same')(conv9)
conv9 = Conv2D(width, (3, 3), activation='relu',padding = 'same')(conv9)

conv10 = Conv2D(2, (3, 3), activation='relu',padding = 'same')(conv9)
conv10 = Conv2D(2, (1, 1), activation='softmax')(conv10)
model = Model(inputs = inputs, outputs = conv10)
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',metrics=['acc'])
history = model.fit(img,one_hot_lab, batch_size = 5,epochs = 50, shuffle = True)