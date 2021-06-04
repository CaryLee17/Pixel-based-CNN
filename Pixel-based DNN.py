import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
from keras.models import Model
from keras.layers import *
import os
from PIL import Image
from keras import optimizers
path = r'D:\研究生\影像解译\色楞'
os.chdir(path)
fn = os.listdir(os.path.join(path,'label5'))
img = np.zeros((len(fn),512,512,7))
lab = np.zeros((len(fn),512,512,2))
for i,num in zip(fn,range(len(fn))):
    img[num] = tif.imread(path+'\\crop\\'+i.split('.')[0]+'.tif').transpose([1,2,0])/2**16
    im = np.array(Image.open(path+'\\label5\\'+i))
    lab[num,:,:,0][np.where(im==1)] = 1
    lab[num,:,:,1][np.where(im==2)] = 1
index = np.where(lab[:,:,:,0]+lab[:,:,:,1] != 0)
labels = lab[index[0],index[1],index[2],:]
images = img[index[0],index[1],index[2],:]
la = labels.copy()
la[:,1]=0
im = images.copy()
index = np.where(la[:,0]==1)
la = la[index]
im = im[index]
for i in range(10):
    labels = np.concatenate((labels, la))
    images = np.concatenate((images,im))
del la, im
import numpy as np
from keras import optimizers
from keras.models import Model
from keras.layers import *
def DNN():
    inputs = Input((7,))
    x = Dense(16, activation = 'relu')(inputs)
    x = Dense(64, activation = 'relu')(x)
    x = Dense(128, activation = 'relu')(x)
    outputs = Dense(2,activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
model = DNN()
model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()
history = model.fit(images, labels, batch_size=1000,epochs = 50,shuffle = True)