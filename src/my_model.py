'''
author: Raisa
date 21-11-19
A new CNN model which is given in the video lecture
Used a little data augmentation technique 
'''

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from tqdm import tqdm
import os
import pandas as pd 
import config

#reading the label file
train = pd.read_csv('E:/CSE/12th_semester/CIFAR-10-Project/src/trainLabels.csv')

#loading the train dataset
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('E:/CSE/12th_semester/CIFAR-10-Project/data/train/'+train['id'][i].astype('str')+'.png', target_size=(28,28,3))
    img = image.img_to_array(img)  #convert to numpy array
    img = img/255
    train_image.append(img) 
X = np.array(train_image)
#on hot encoding
y=train['label'].values
y[y == 'airplane'] = 0
y[y == 'automobile'] = 1
y[y == 'bird'] = 2
y[y == 'cat'] = 3
y[y == 'deer'] = 4
y[y == 'dog'] = 5
y[y == 'frog'] = 6
y[y == 'horse'] = 7
y[y == 'ship'] = 8
y[y == 'truck'] = 9

#convert the labels into categorical  
num_classes=10
y = np_utils.to_categorical(y,num_classes)
#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# CNN model
"""model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),input_shape=(28,28,3),padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(32, kernel_size=(3, 3),input_shape=(28,28,3),padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.20))

model.add(Conv2D(64, kernel_size=(3, 3),input_shape=(28,28,3),padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, kernel_size=(3, 3),input_shape=(28,28,3),padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.20))

model.add(Flatten())
model.add(Dense(384,kernel_regularizer=keras.regularizers.l2(l=0.01)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.30))
model.add(Dense(num_classes, activation='softmax'))"""

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# compile model
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
# create data generator
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
# prepare iterator
it_train = datagen.flow(X_train, y_train, batch_size=200)
# fit model
steps = int(X_train.shape[0] / 64)
history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=100, validation_data=(X_test, y_test), verbose=2)
	

#model_checkpoint_dir=os.path.join(config.checkpoint_path(),"baseline.h5")
model_checkpoint_dir=os.path.join("E:/CSE/12th_semester/CIFAR-10-Project/checkpoint/","baseline.h5")
#complie the model
#model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#training the model
'''model.fit(X_train, y_train, batch_size=200, epochs=100, verbose=2, callbacks=[EarlyStopping(monitor='val_loss',
            patience=20, verbose=2, mode='auto'),
            ModelCheckpoint(model_checkpoint_dir, monitor='val_loss', verbose=2, save_best_only=True,
            save_weights_only=False, mode='auto', period=1)],shuffle=True,validation_data=(X_test, y_test))'''
#read and store all test image
nb_img=50000
start=1
for part in range(0,6):
    if not(part==0): 
        start=start+nb_img
    last=start+nb_img

    print("iteration from",start," to ",last)
    test_image = []
    for i in tqdm(range(start,last)):
        img = image.load_img('E:/CSE/12th_semester/CIFAR-10-Project/data/test/'+str(i)+'.png', target_size=(28,28,3))
        img = image.img_to_array(img)
        img = img/255
        test_image.append(img)
    y = np.array(test_image)
    # making predictions
    prediction = model.predict_classes(y, batch_size=200, verbose=2)
    # creating submission file
    sample = pd.read_csv('E:/CSE/12th_semester/CIFAR-10-Project/src/submission.csv')
    sample['label'] = prediction
    sample.to_csv('E:/CSE/12th_semester/CIFAR-10-Project/output/sample_cnn_'+str(part)+'.csv', header=True, index=False)
print("Complete")