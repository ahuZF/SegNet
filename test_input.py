import numpy as np
from PIL import  Image
np.set_printoptions(threshold=np.inf)

'''img = Image.open("./dataset2/dataset2/jpg/9.jpg" )
print(img.size)
img = img.resize((416, 416))
#img.save("./img_out/zf_1111.jpg" )
img = Image.open("./dataset2/dataset2/png/9.png" )
img = img.resize((416, 416))
img = np.array(img)
print(img)'''
#print(img)


img = Image.open(r".\dataset2\dataset2\png\9.png" )
img = img.resize((416, 416))
img = np.array(img)
#seg_labels = np.zeros((int(HEIGHT/2),int(WIDTH/2),NCLASSES))
seg_labels = np.zeros((416, 416, 2))##创造一个空白img数组
for c in range(2):
    seg_labels[: , : , c ] = (img[:,:,0] == c ).astype(int)##给数组赋值
seg_labels = np.reshape(seg_labels, (-1,2))
print(seg_labels.shape)
