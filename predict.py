from unet.train import segNet_decoder
from PIL import Image
import numpy as np
import random
import copy
import os

np.set_printoptions(threshold=np.inf)

class_colors = [[0,0,0],[0,255,0]]
NCLASSES = 2
HEIGHT = 416
WIDTH = 416


model = segNet_decoder()
#model.load_weights("logs/ep008-loss0.194-val_loss0.164.h5")
model.load_weights("logs/last1.h5")

img = Image.open("./dataset2/dataset2/jpg/9.jpg" )
#img = Image.open(r"C:\Users\123\Desktop\2.png" )
old_img = copy.deepcopy(img)
orininal_h = np.array(img).shape[0]
orininal_w = np.array(img).shape[1]

img = img.resize((WIDTH, HEIGHT))
img = np.array(img)
img = img / 255
img = img.reshape(-1, HEIGHT, WIDTH, 3)
pr = model.predict(img)[0]
print(pr.size)


pr = pr.reshape((HEIGHT, WIDTH, NCLASSES)).argmax(axis=-1)
print('argmax:',pr)

seg_img = np.zeros((HEIGHT,WIDTH, 3))### 产生一个空的3通道img
colors = class_colors

for c in range(NCLASSES):
    seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
    seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
    seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))

image = Image.blend(old_img, seg_img, 0.3)
image.save("./img_out/zf_111.jpg" )