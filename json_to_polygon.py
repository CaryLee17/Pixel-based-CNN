import json
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
import os
from PIL import Image
width,height = 512, 512
path = r"D:\研究生\影像解译\色楞\Linear"
os.chdir(path)
file = os.listdir(path+'\\json')
try:
    os.mkdir("label5")
except:
    pass
os.chdir(os.path.join(path,'label5'))
def json_to_png(file):
    for count in file:
        lab = np.zeros((width,height))
        jsonfile = json.load(open(os.path.join(path+'\\json', count)))
        for i in jsonfile['instances']:
            x,y = polygon(i['points'][::2],i['points'][1::2])
            if i['className']=='water':
                lab[y-1,x-1]=1
            elif i['className'] == 'others':
                lab[y-1,x-1]=2
        Image.fromarray(lab.astype('uint8')).save(count.split('.json')[0])
json_to_png(file)