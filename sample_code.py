#author Raisa
#date 31-10-19
#found a simple code on a blogpage and tried just to see it run. didn't fully copy pasted.
#blog link: https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pandas as pd 
#reading the label file
train = pd.read_csv('trainLabels.csv')
#set grayscale as False
train_image = []
#loading the train dataset
for i in tqdm(range(train.shape[0])):
    img = image.load_img('train/'+train['id'][i].astype('str')+'.png', target_size=(28,28,3))
    img = image.img_to_array(img)
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
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))
#read and store all test image

#first_index=1
#second_index=50000
#first_index =50001
#second_index = 100000
#first_index=100001  
#second_index=150000
#first_index=150001
#second_index=200000
#first_index=200001
#second_index=250000
#first_index=250001
#second_index=300000

print("iteration from",first_index)
test_image = []
for i in tqdm(range(first_index,(second_index+1))):
    #img = image.load_img('test/'+test['id'][i].astype('str')+'.png', target_size=(28,28,3))
    img = image.load_img('test/'+str(i)+'.png', target_size=(28,28,3))
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
y = np.array(test_image)
# making predictions
prediction = model.predict_classes(y)
# creating submission file
sample = pd.read_csv('submission.csv')
sample['label'] = prediction
#sample.to_csv('sample_cnn1.csv', header=True, index=False)
#sample.to_csv('sample_cnn2.csv', header=True, index=False)
#sample.to_csv('sample_cnn3.csv', header=True, index=False)
#sample.to_csv('sample_cnn4.csv', header=True, index=False)
#sample.to_csv('sample_cnn5.csv', header=True, index=False)
#sample.to_csv('sample_cnn6.csv', header=True, index=False)