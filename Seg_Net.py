from keras.layers  import *
from keras.models  import *
from pool_indices import MaxUnpooling2D,MaxPoolingWithArgmax2D
import  numpy as np

img_w=416
img_h=416
inputs = Input((img_w, img_h, 3))
def segNet_enconder():
    ##416
    x = Conv2D(64, (3, 3), padding='same', use_bias=True, strides=[1, 1],name='block1_conv1')(inputs)##默认为"channels_last"， (batch, height, width, channels)
    #x=BatchNormalization(axis=1)(x)
    x=Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', use_bias=True, strides=[1, 1], name='block1_conv2')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    #x=MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='valid')(x)
    #f1 = x ##208
    x, f1 = MaxPoolingWithArgmax2D(name='block1_pool')(x)
    ##### 第一层

    ##208
    x = Conv2D(128, (3, 3), padding='same', use_bias=True, strides=[1, 1],name='block2_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', use_bias=True, strides=[1, 1],name='block2_conv2')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    #x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x)
    #f2 = x ##104
    x, f2 = MaxPoolingWithArgmax2D(name='block2_pool')(x)
    ##### 第二层

    ##104
    x = Conv2D(256, (3, 3), padding='same', use_bias=True, strides=[1, 1], name='block3_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', use_bias=True, strides=[1, 1], name='block3_conv2')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', use_bias=True, strides=[1, 1], name='block3_conv3')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    #x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x)
    #f3 = x ##52
    x, f3 = MaxPoolingWithArgmax2D( name='block3_pool')(x)
    ##### 第三层

    ##52
    x = Conv2D(512, (3, 3), padding='same', use_bias=True, strides=[1, 1],name='block4_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', use_bias=True, strides=[1, 1],name='block4_conv2')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', use_bias=True, strides=[1, 1],name='block4_conv3')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    #x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x)
    #f4 = x ##26
    x, f4 = MaxPoolingWithArgmax2D(name='block4_pool1')(x)

    ##### 第四层
    ##26
    x = Conv2D(512, (3, 3), padding='same', use_bias=True, strides=[1, 1],name='block5_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', use_bias=True, strides=[1, 1],name='block5_conv2')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', use_bias=True, strides=[1, 1],name='block5_conv3')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    #x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid')(x) ##13
    #f5 = x
    x, f5 = MaxPoolingWithArgmax2D(name='block5_pool1')(x)
    ##### 第五层

    Vgg_streamlined = Model(inputs=inputs, outputs=x)

    # 加载vgg16的预训练权重
    Vgg_streamlined.load_weights(r".\logs\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    return x,[f1,f2,f3,f4,f5]

def segNet_decoder():
    y,f=segNet_enconder()
    #x=f[4] ## f5 13
    #x=UpSampling2D(size=(2, 2))(x)##26

    unpool_1 = MaxUnpooling2D()([y, f[4]])
    x = Conv2D(512, (3, 3), padding='same', use_bias=False, strides=[1, 1])(unpool_1)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', use_bias=False, strides=[1, 1])(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', use_bias=False, strides=[1, 1])(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    #### 第六层
    #x = add([x, f[3]])
    #x = UpSampling2D(size=(2, 2))(x)#52
    unpool_2 = MaxUnpooling2D()([x, f[3]])
    x = Conv2D(512, (3, 3), padding='same', use_bias=False, strides=[1, 1])(unpool_2)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', use_bias=False, strides=[1, 1])(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', use_bias=False, strides=[1, 1])(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    #### 第七层
    #x = add([x, f[2]])
    #x = UpSampling2D(size=(2, 2))(x)#104
    unpool_3 = MaxUnpooling2D()([x, f[2]])
    x = Conv2D(256, (3, 3), padding='same', use_bias=False, strides=[1, 1])(unpool_3)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', use_bias=False, strides=[1, 1])(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', use_bias=False, strides=[1, 1])(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    #### 第八层
    #x = add([x, f[1]])
    #x = UpSampling2D(size=(2, 2))(x)#208
    unpool_4 = MaxUnpooling2D()([x, f[1]])
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, strides=[1, 1])(unpool_4)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', use_bias=False, strides=[1, 1])(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    #### 第九层
    #x = add([x, f[0]])
    #x = UpSampling2D(size=(2, 2))(x)#416
    unpool_5 = MaxUnpooling2D()([x, f[0]])
    x = Conv2D(64, (3, 3), padding='same', use_bias=False, strides=[1, 1])(unpool_5)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(2, (3, 3), padding='same', use_bias=False, strides=[1, 1])(x)
    x = BatchNormalization(axis=1, name='conv13_bn')(x)
    x = Activation('relu')(x)
    x=  Reshape((416*416,-1))(x)
    #x = Reshape((-1, 2))(x)  #变成两列同上句
    o = Softmax()(x)
    model = Model(inputs=inputs, outputs=o)
    return model
    #### 第十层









