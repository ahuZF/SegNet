import train
from keras.layers import *
from keras.models import *
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import numpy as np
from keras import backend as K
from keras.optimizers import Adam

HEIGHT=416
WIDTH=416
NCLASSES=2
def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = Image.open(r".\dataset2\dataset2\jpg" + '/' + name)
            img = img.resize((WIDTH,HEIGHT))
            img = np.array(img)### 变成一维向量
            img = img/255
            X_train.append(img)

            name = (lines[i].split(';')[1]).replace("\n", "")
            # 从文件中读取图像
            img = Image.open(r".\dataset2\dataset2\png" + '/' + name)
            img = img.resize((WIDTH, HEIGHT))
            img = np.array(img)
            #seg_labels = np.zeros((int(HEIGHT/2),int(WIDTH/2),NCLASSES))
            seg_labels = np.zeros((HEIGHT, WIDTH, NCLASSES))##创造一个空白img数组
            for c in range(NCLASSES):
                seg_labels[: , : , c ] = (img[:,:,0] == c ).astype(int)##给数组赋值
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))##给标签的第三维度分割成one-hot类型[0,1]
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i+1) % n
        yield (np.array(X_train),np.array(Y_train))

def loss(y_true, y_pred):
    crossloss = K.binary_crossentropy(y_true,y_pred)
    # 这个乘除是为了求平均……相当于每个像素点的交叉熵。
    loss =K.sum(crossloss)/HEIGHT/WIDTH
    return loss


if __name__ == "__main__":
    log_dir = "logs/"
    # 获取model
    model = train.segNet_decoder()
    model.summary()
    with open(r".\dataset2\dataset2\train.txt", "r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 保存的方式，3世代保存一次
    checkpoint_period = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        period=3
    )
    # 学习率下降的方式，val_loss3次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 交叉熵
    model.compile(loss=loss,
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])
    batch_size = 4
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=20,
                        initial_epoch=0,
                        callbacks=[checkpoint_period, reduce_lr])

    model.save_weights(log_dir + 'last1.h5')