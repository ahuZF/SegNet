import cv2
import random
import numpy as np
import os
import argparse
from keras.preprocessing.image import img_to_array
import Seg_Net
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import unet_train

np.set_printoptions(threshold=np.inf)
model= Seg_Net.segNet_decoder()
#model.summary()
img=cv2.imread(r'.\src\484.png')
#cv2.imshow('build',img)
#cv2.waitKey(0)
img=img_to_array(img)
img=np.reshape(img,(1,256,256,3))
model.load_weights(r'.\logs\segnet_last1.h5')
pr=model.predict(img)
#print(pr.shape)
#print(pr)
pr=np.argmax(pr,axis=-1)
#print(pr.shape)
#print(pr)
pred = pr.reshape((256,256)).astype(np.uint8)
cv2.imwrite(r'.\img_out\segnet_484.png',pred)
#print(pred)
'''img = img.resize((256, 256))
img = np.array(img)
img = img.reshape(-1, 256, 256, 3)
model=unet_train.unet()
model.load_weights(r'.\logs\model.h5')
pr=model.predict(img)
print(pr)'''
