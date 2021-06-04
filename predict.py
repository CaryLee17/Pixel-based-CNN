import matplotlib.pyplot as plt
import tifffile as tif
import os
import numpy as np
from keras.models import load_model
from PIL import Image
from osgeo import gdal
fn = '131027'
os.chdir(r'D:\研究生\影像解译\色楞')
model = load_model('landsat_dnn.h5')
x = tif.imread(fn+'.tif')/2**16
y = np.zeros((x.shape[0]+6,x.shape[1]+6,x.shape[2]))
y[3:-3,3:-3,:] = x
del x
def predict(y,model,sz = 512, im_sz=512,k=0):
    pre = np.zeros((y.shape[0]-k*2,y.shape[1]-k*2,2))
    for i in range(0,y.shape[0],im_sz):
        for j in range(0,y.shape[1],im_sz):
            if min(i+sz,y.shape[0]) == y.shape[0] or min(j+sz, y.shape[1]) == y.shape[1]:
                break
            else:
                pre[i:i+im_sz, j:j+im_sz] = model.predict(y[i:i+sz, j:j+sz,:][np.newaxis,:])
    for i in range(0, y.shape[0], im_sz):
        if min(i+sz, y.shape[0]) == y.shape[0]:
            break
        else:
            pre[i:i+im_sz,-im_sz:,:] = model.predict(y[i:i+sz,-sz:,:][np.newaxis,:])
    for j in range(0, y.shape[1], im_sz):
        if min(j+sz, y.shape[1]) == y.shape[1]:
            break
        else:
            pre[-im_sz:,j:j+im_sz,:] = model.predict(y[-sz:,j:j+sz,:][np.newaxis,:])
    pre[-im_sz:,-im_sz:,:] = model.predict(y[-sz:,-sz:,:][np.newaxis,:])
    return pre
pre = predict(y,model,sz=518,k=3)
p = np.argmin(pre,axis=-1)
Image.fromarray((p*255).astype('uint8')).save(r'D:\研究生\影像解译\色楞\结果图\DNN'+'\\'+fn+'.png')
def write_prj(tif_file, mask_file):
    driver = gdal.GetDriverByName("GTiff")
    im = gdal.Open(r'D:\研究生\影像解译\色楞'+'\\'+tif_file)
    print(r'D:\研究生\影像解译\色楞'+'\\'+tif_file)
    out = driver.Create(r'D:\研究生\影像解译\色楞\结果图\test'+'\\'+mask_file.split('.')[0]+'.tif', im.GetRasterBand(1).XSize, im.GetRasterBand(1).YSize, 1, im.GetRasterBand(1).DataType)
    index = np.where(im.ReadAsArray()[4] == 0)
    img = np.array(Image.open(r'D:\研究生\影像解译\色楞\结果图\test'+'\\'+mask_file))
    img[index] = 2
    out.GetRasterBand(1).WriteArray(img)
    out.SetGeoTransform(im.GetGeoTransform())
    out.SetProjection(im.GetProjection())
    out.FlushCache()
    del out
file = {'img':["131027.tif",'132027.tif','133027.tif'],
        'mask':['131027.png','132027.png','133027.png']}
for i, j in zip(file['img'],file['mask']):
    write_prj(i,j)