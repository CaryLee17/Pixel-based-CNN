import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
from keras.models import Model
from keras.layers import *
import os
from PIL import Image
from keras import optimizers
from keras.models import load_model
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
print(img.shape)
im = np.zeros((img.shape[0],img.shape[1]+6, img.shape[2]+6,img.shape[3]))
im[:,3:-3,3:-3,:] = img
index = np.where(lab[:,:,:,0]+lab[:,:,:,1] != 0)
images = np.zeros((index[0].shape[0],7,7,7))
labels = lab[index[0],index[1],index[2],:]
for i in range(images.shape[0]):
    images[i] = im[index[0][i],index[1][i]:index[1][i]+7,index[2][i]:index[2][i]+7,:]
print(images.shape)
im = images.copy()
la = labels.copy()
la[:,1]=0
index = np.where(la[:,0]==1)
la = la[index]
im = im[index]
for i in range(10):##
    images = np.concatenate((images,im))
    labels = np.concatenate((labels,la))
del im, la
def CNN():
    inputs = Input((7,7,7))
    x = Conv2D(16, (3, 3), activation = 'relu')(inputs)
    x = Conv2D(32, (3, 3), activation = 'relu')(x)
    x = concatenate([x,inputs[:,2:-2,2:-2,:]])
    x = Conv2D(64, (3, 3), activation = 'relu')(x)
    x = concatenate([x,inputs[:,3:-3,3:-3,:]])
    x = Conv2D(128, (1, 1), activation = 'relu')(x)
    outputs = Conv2D(2,(1,1),activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
model = CNN()
model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()
labels = labels.reshape((len(labels),1,1,2))
history = model.fit(images, labels, batch_size=1000,epochs = 50, shuffle = True)
model.save('model.h5')