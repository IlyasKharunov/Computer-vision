from keras import layers, callbacks
from keras.models import Sequential 
import numpy as np
from os.path import join
from os import listdir
from skimage.transform import resize
from skimage.io import imread

px = 128
def train_detector(train_gt, train_img_dir, fast_train=True):
    check = callbacks.ModelCheckpoint('facepoints_model.hdf5', save_best_only = True)
    picnum = len(train_gt)
    if fast_train==False:
        dataa = np.empty((picnum*4,px,px,3))
        labells = np.empty((picnum*4,28))
    else:
        dataa = np.empty((picnum,px,px,3))
        labells = np.empty((picnum,28))
    valueff = np.empty((28))
    valuef = np.empty((28))
    i = 0
    for key,value in train_gt.items():
        im = imread(join(train_img_dir,key))
        if len(im.shape) == 2:
            im = np.dstack((im,im,im))
        xreshape = px/im.shape[1]
        yreshape = px/im.shape[0]
        imr = resize(im,(px,px,3),anti_aliasing = False)
        dataa[i,:,:,:] = imr
        value[::2]= value[::2]*xreshape 
        value[1::2] = value[1::2]*yreshape
        labells[i,:] = value
        i+=1
        if fast_train == False:
            dataa[i,:,:,:] = np.fliplr(imr)
            valuef[1::2] = value[1::2]
            valuef[::2] = imr.shape[1] - value[::2]        
            valueff[0:2] = valuef[6:8]
            valueff[2:4] = valuef[4:6]
            valueff[4:6] = valuef[2:4]
            valueff[6:8] = valuef[0:2]
            valueff[8:10] = valuef[18:20]
            valueff[10:12] = valuef[16:18]
            valueff[12:14] = valuef[14:16]
            valueff[14:16] = valuef[12:14]
            valueff[16:18] = valuef[10:12]
            valueff[18:20] = valuef[8:10]
            valueff[20:22] = valuef[20:22]
            valueff[22:24] = valuef[26:28]
            valueff[24:26] = valuef[24:26]
            valueff[26:28] = valuef[22:24]
            labells[i,:] = valueff
            print(i)
            i+=1
            dataa[i,:,:,:] = np.flipud(imr)
            valuef[::2] = value[::2]
            valuef[1::2] = imr.shape[0] - value[1::2]        
            valueff[0:2] = valuef[6:8]
            valueff[2:4] = valuef[4:6]
            valueff[4:6] = valuef[2:4]
            valueff[6:8] = valuef[0:2]
            valueff[8:10] = valuef[18:20]
            valueff[10:12] = valuef[16:18]
            valueff[12:14] = valuef[14:16]
            valueff[14:16] = valuef[12:14]
            valueff[16:18] = valuef[10:12]
            valueff[18:20] = valuef[8:10]
            valueff[20:22] = valuef[20:22]
            valueff[22:24] = valuef[26:28]
            valueff[24:26] = valuef[24:26]
            valueff[26:28] = valuef[22:24]
            labells[i,:] = valueff
            print(i)
            i+=1
            dataa[i,:,:,:] = np.transpose(imr,(1,0,2))
            valuef[::2] = value[1::2]
            valuef[1::2] = value[::2]        
            valueff[0:2] = valuef[6:8]
            valueff[2:4] = valuef[4:6]
            valueff[4:6] = valuef[2:4]
            valueff[6:8] = valuef[0:2]
            valueff[8:10] = valuef[18:20]
            valueff[10:12] = valuef[16:18]
            valueff[12:14] = valuef[14:16]
            valueff[14:16] = valuef[12:14]
            valueff[16:18] = valuef[10:12]
            valueff[18:20] = valuef[8:10]
            valueff[20:22] = valuef[20:22]
            valueff[22:24] = valuef[26:28]
            valueff[24:26] = valuef[24:26]
            valueff[26:28] = valuef[22:24]
            labells[i,:] = valueff
            print(i)
            i+=1
    for j in range (3):
        mean = np.mean(dataa[:,:,:,j],axis = 0)
        std = np.std(dataa[:,:,:,j],axis = 0)
    for i in range(len(dataa)):
        dataa[i,:,:,j] = (dataa[i,:,:,j]-mean)/std
    model = Sequential()
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(24,(5,5),data_format="channels_last",activation='relu',input_shape = (px,px,3)))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.SpatialDropout2D(0.2))
    model.add(layers.Conv2D(36,(5,5),data_format="channels_last",activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.SpatialDropout2D(0.2))
    model.add(layers.Conv2D(48,(5,5),data_format="channels_last",activation='relu'))
    model.add(layers.MaxPooling2D(padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.SpatialDropout2D(0.2))
    model.add(layers.Conv2D(128,(3,3),data_format="channels_last",activation='relu'))
    model.add(layers.MaxPooling2D(padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.SpatialDropout2D(0.2))
    model.add(layers.Conv2D(128,(3,3),data_format="channels_last",padding = 'same',activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.SpatialDropout2D(0.2))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(28))   
    model.compile(optimizer='Adam',
              loss='mse',
              metrics=['accuracy'])
    if fast_train == True:
        model.fit(dataa, labells, batch_size=1000, epochs=1)
        model.save('facepoints_model.hdf5')
    else:
        model.fit(dataa, labells, batch_size=128, epochs=800,validation_split = 0.2, callbacks = ([check]))
    
def detect(model, test_img_dir):
    imlist = listdir(test_img_dir)
    imnum = len(imlist)
    xreshape = np.empty((imnum,14))
    yreshape = np.empty((imnum,14))
    dataa = np.empty((imnum,px,px,3))
    for i in range(imnum):
        print(i)
        im = imread(join(test_img_dir,imlist[i]))
        if len(im.shape) == 2:
            im = np.dstack((im,im,im))
        xreshape[i,:] = im.shape[1]/px
        yreshape[i,:] = im.shape[0]/px
        dataa[i,:,:,:] = resize(im,(px,px,3),anti_aliasing = False)
    for j in range (3):
        mean = np.mean(dataa[:,:,:,j],axis = 0)
        std = np.std(dataa[:,:,:,j],axis = 0)
    for i in range(len(dataa)):
        print(i)
        dataa[i,:,:,j] = (dataa[i,:,:,j]-mean)/std
    result = {}
    resultar = model.predict(dataa, batch_size=64)
    resultar[:,::2] = resultar[:,::2]*xreshape
    resultar[:,1::2] = resultar[:,1::2]*yreshape
    for i in range(imnum):
        result[imlist[i]] = resultar[i,:]
    return result;