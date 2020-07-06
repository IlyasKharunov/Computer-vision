from keras import layers, callbacks, applications, utils
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from os.path import join
from os import listdir
from skimage.transform import resize
from skimage.io import imread
from sklearn.model_selection import train_test_split
px = 224
seed = 226
mnum = 3

def train_classifier(train_gt, train_img_dir, fast_train=True):
    picnum = len(train_gt)
    dataa = np.empty((picnum,px,px,3))
    labells = np.empty((picnum))
    i = 0
    for key,value in train_gt.items():
        im = imread(join(train_img_dir,key))
        if len(im.shape) == 2:
            im = np.dstack((im,im,im))
        imr = resize(im,(px,px,3),mode = 'reflect')
        dataa[i,:,:,:] = imr
        labells[i] = value
        i+=1
        print(i)
    if fast_train == False:
        d_tr, d_te, l_tr, l_te = train_test_split(dataa, labells, test_size = 0.2, random_state = seed, stratify = labells)
        labels = utils.to_categorical(l_tr)
        l_te = utils.to_categorical(l_te)
    else:
        labels = utils.to_categorical(labells)
    datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.2,height_shift_range=0.2, horizontal_flip = True)
    check = callbacks.ModelCheckpoint('birds_model.hdf5', monitor='val_acc', save_best_only = True)
    base_model = applications.xception.Xception(include_top=False, weights = 'imagenet')
    x = base_model.layers[-1].output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(500, activation = 'relu')(x)
    predictions = layers.Dense(50, activation = 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    #model1 = load_model('birds_model.hdf5')
    #model.set_weights(model1.get_weights())
    if fast_train == True:
        print('okk')
    else:
        model.compile(optimizer = Adam(lr = 0.000005), loss='categorical_crossentropy', metrics = ['accuracy'])
        model.fit_generator(datagen.flow(d_tr, labels, batch_size=32),steps_per_epoch=len(d_tr)/32,
                            epochs=1000,validation_data = (d_te,l_te), callbacks = ([check]))

def classify(model, test_img_dir):
    imlist = listdir(test_img_dir)
    imnum = len(imlist)
    dataa = np.empty((imnum,px,px,3))
    for i in range(imnum):
        print(i)
        im = imread(join(test_img_dir,imlist[i]))
        if len(im.shape) == 2:
            im = np.dstack((im,im,im))
        dataa[i,:,:,:] = resize(im,(px,px,3),anti_aliasing = False)
    '''for j in range (3):
        mean = np.mean(dataa[:,:,:,j],axis = 0)
        std = np.std(dataa[:,:,:,j],axis = 0)
    for i in range(len(dataa)):
        print(i)
        dataa[i,:,:,j] = (dataa[i,:,:,j]-mean)/std'''
    result = {}
    resultar = model.predict(dataa, batch_size=32)
    resullt = np.empty((imnum))
    for i in range(imnum):
        resullt[i] = np.argmax(resultar[i,:])
        result[imlist[i]] = resullt[i]
    return result;