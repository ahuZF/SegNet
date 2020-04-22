import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
#np.set_printoptions(threshold=np.inf)
path=r'./label/3.png'
train_label = []
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#label = load_img(filepath + 'label/' + url, grayscale=True)#True
img = img_to_array(img)###变成浮点数
print(img.shape)
print(img)
#img=np.reshape(img,(256,256))

seg_labels = np.zeros((256, 256, 2))##创造一个空白img数组
for c in range(2):
    seg_labels[: , : , c ] = (img[:,:,0] == c ).astype(int)##给数组赋值
seg_labels = np.reshape(seg_labels, (-1,2))

print(seg_labels.shape)
print(seg_labels)
#img=np.reshape(img,(-1,2))
#print(img)
#print(img.shape)
#train_label.append(img)
#train_label=np.array(train_label)
#label=np.reshape(train_label,(-1,2))


'''seg_labels = np.zeros((416, 416, 2))##创造一个空白img数组
for c in range(2):
    seg_labels[: , : , c ] = (img[:,:,1] == c ).astype(int)##给数组赋值
seg_labels = np.reshape(seg_labels, (-1,2))'''