import numpy as np
import os
from PIL import Image
import tifffile as tif
os.chdir("D:\研究生\影像解译\色楞")
try:
    os.mkdir("crop")
except:
    pass
try:
    os.mkdir("Linear")
except:
    pass
    def img_read_gdal(path = r"D:\研究生\影像解译\色楞\132027"):
    from osgeo import gdal
    x = gdal.Open(os.path.join(path,os.listdir(path)[0]))
    driver = gdal.GetDriverByName('GTiff')
    img = driver.Create("132027.tif",x.GetRasterBand(1).XSize,x.GetRasterBand(1).YSize,7,x.GetRasterBand(1).DataType)
    img.SetProjection(x.GetProjection())
    img.SetGeoTransform(x.GetGeoTransform())
    num = 1
    for i in os.listdir(path):
        print(os.path.join(path,i))
        img.GetRasterBand(num).WriteArray(gdal.Open(os.path.join(path,i)).ReadAsArray())
        num+=1
    img.FlushCache()
    del img,x
    img = gdal.Open(os.path.join(path,"ttt.tif")).ReadAsArray()
    return img
def img_read_files(path = r"D:\研究生\影像解译\色楞\131027"):
    img = tif.imread(os.path.join(path,os.listdir(path)[0]))[np.newaxis,:]
    for i in os.listdir(path)[1:]:
        img = np.concatenate((img,tif.imread(os.path.join(path, i))[np.newaxis,:]),axis=0)
    tif.imsave('131027.tif',img)
    return img
def crop_action(x,count):
    tif.imsave(r"crop\%03d.tif" % count,x)
    if x.sum()==0:
        pass
    else:
        x = x[4:1:-1,:,:]/2.**16
        x =x.transpose([1,2,0])
        x = (x-x.min())/(x.max()-x.min())*255
        x = Image.fromarray(np.uint8(x))
        x.save(r"Linear\%03d.png" % count)
print(">>>>>>>>>>>>Crop Processing>>>>>>>>>>>>")
def crop_img(img):
    count = 1
    for i in range(0,img.shape[1],512):
        for j in range(0, img.shape[2], 512):
            #裁切正中间区域
            if min(i+512,img.shape[1]) == img.shape[1] or min(j+512, img.shape[2]) == img.shape[2]:
                break
            else:
                x = img[:,i:i+512, j:j+512]
                crop_action(x,count)
                count+=1
    for i in range(0, img.shape[1], 512):
        if min(i+512, img.shape[1]) == img.shape[1]:
            break
        else:
            x = img[:,i:i+512,-512:]
            crop_action(x,count)
            count+=1
    for j in range(0, img.shape[2], 512):
        if min(j+512, img.shape[2]) == img.shape[2]:
            break
        else:
            x = img[:,-512:,j:j+512]
            crop_action(x,count)
            count+=1
    x = img[:,-512:,-512:]
    crop_action(x,count)
    print("Done!")
crop_img(img_read_files())