#author Raisa
#date 31-10-19
#found a simple code on a blogpage and tried just to see it run. didn't fully copy pasted.
#blog link: https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pandas as pd 
train = pd.read_csv('trainLabels.csv')
#set grayscale as False
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('train/'+train['id'][i].astype('str')+'.png', target_size=(28,28,1), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img) #loading the train dataset
X = np.array(train_image)
#on hot encoding
y=train['label'].values
"""for x in y:
    if x=='airplane':
        y[x]=0
    elif x=='automobile':
        y[x]=1
    elif x=='bird':
        y[x]=2
    elif x=='cat':
        y[x]=3
    elif x=='deer':
        y[x]=4
    elif x=='dog':
        y[x]=5
    elif x=='frog':
        y[x]=6
    elif x=='horse':
        y[x]=7
    elif x=='ship':
        y[x]=8
    elif x=='truck':
        y[x]=9"""
    
#print(type(y))
#keras.utils.to_categorical(y, num_classes=None, dtype='float32')
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
#complie the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#training the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
#read and store all test image
test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img('test/'+test['id'][i].astype('str')+'.png', target_size=(28,28,3), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)
# making predictions
prediction = model.predict_classes(test)
# creating submission file
sample = pd.read_csv('sample_submission.csv')
sample['label'] = prediction
sample.to_csv('sample_cnn.csv', header=True, index=False)