#coding=utf-8
#from keras import backend as K  #
#K.set_image_dim_ordering('th')#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np  
from keras.models import Sequential  
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Input  
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint  
from sklearn.preprocessing import LabelEncoder  
from keras.models import Model
from keras.layers.merge import concatenate
from PIL import Image  
import matplotlib.pyplot as plt  
import cv2
import random
import os
import Seg_Net
from tqdm import tqdm
seed = 7  
np.random.seed(seed)  
classes = [0. ,  1.,  2.,   3.  , 4.]
#data_shape = 360*480  
img_w = 256  
img_h = 256
labelencoder = LabelEncoder()
labelencoder.fit(classes)

#filepath =r'G:\深度学习\train\train'
def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)#path
    else:
        img = cv2.imread(path)#path
        img = np.array(img,dtype="float") / 255.0
    return img



def get_train_val(val_rate = 0.2):
    train_url = []    
    train_set = []
    val_set  = []
    for pic in os.listdir(filepath + 'src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set

# data for training  
def generateData(batch_size,data=[]):  
    #print 'generateData...'
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img(filepath + 'src/' + url)
            img = img_to_array(img)  
            train_data.append(img)  
            label = load_img(filepath + 'label/' + url, grayscale=True) 
            label = img_to_array(label)

            seg_labels = np.zeros((256, 256, 2))  ##创造一个空白img数组
            for c in range(2):
                seg_labels[:, :, c] = (label[:, :, 0] == c).astype(int)  ##给数组赋值
            seg_labels = np.reshape(seg_labels, (-1, 2))
            train_label.append(seg_labels)


            #train_label.append(label)
            if batch % batch_size==0: 
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)  
                train_label = np.array(train_label)  
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  
 
# data for validation 
def generateValidData(batch_size,data=[]):  
    #print 'generateValidData...'
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_img(filepath + 'src/' + url)
            img = img_to_array(img)  
            valid_data.append(img)  
            label = load_img(filepath + 'label/' + url, grayscale=True)#True
            label = img_to_array(label)

            seg_labels = np.zeros((256, 256, 2))  ##创造一个空白img数组
            for c in range(2):
                seg_labels[:, :, c] = (label[:, :, 0] == c).astype(int)  ##给数组赋值
            seg_labels = np.reshape(seg_labels, (-1, 2))
            valid_label.append(seg_labels)

            #label=np.reshape(label,(-1,2))
            #valid_label.append(label)
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label)  
                yield (valid_data,valid_label)  #返回同时下一次运行在从下一句运行，后面三句清空
                valid_data = []  
                valid_label = []  
                batch = 0  
  
  
'''def unet():
    inputs = Input((img_w, img_h,3))#(3,img_w,img_h)

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#dim_ordering="th"

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#dim_ordering="th"

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#dim_ordering="th"

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)#dim_ordering="th"

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)#1
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)#1
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)#1
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)#1
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(1, (1, 1), activation="sigmoid")(conv9)
    #conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model'''


  
def train(args): 
    EPOCHS = 20
    BS = 4  #16
    #model = Seg_Net.segNet_decoder()
    #model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model = unet()
    model= Seg_Net.segNet_decoder()
    model.summary()
    modelcheck = ModelCheckpoint(args["model"],monitor='val_acc',save_best_only=True,save_weights_only=True,mode='max')  #'model'  监视val_acc里面最大的
    callable = [modelcheck]
    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,  
                    validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=callable,max_q_size=1)
    model.save_weights(r'./logs/segnet_last1.h5')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy on Seg-Net ")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(r"./img_out/plot_segnet.png")#args["plot"]


  

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", help="training data's path",#F:/unet/unet_train/
                    default=True)
    ap.add_argument("-m", "--model", required=True,  #unet_buildings20.h5
                    help="path to output model")
    #ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    #help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args


if __name__=='__main__':  
    args = args_parse()
    filepath = args["data"] #['data']
    train(args)  
    #predict()  
