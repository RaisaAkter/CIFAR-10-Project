#author raisa
#this python script is used for experiment different thing and if it goes right then I add them in my main code

"""first_index = 1; second_index = 50000
img_files= [TEST_DATA_DIR + str(i) +  
                    ".png"  for i in range(first_index, (second_index + 1))]

test_data = np.ndarray(((second_index - first_index), IMG_SIZE,  IMG_SIZE, IMG_CHANNEL),
                 dtype = np.float32)

for i, img_file in enumerate(img_files):
        img = Image.open(img_file)
        test_data[i] = np.array(img)"""
from tqdm import tqdm
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

from os import listdir
from PIL import Image as PImage


"""test = pd.read_csv('test.csv')
#print(test.shape[0])
test_image = []
f_index=1
l_index=10000
for i in tqdm(range(f_index,(l_index+1))):
    img = image.load_img('test/'+test['id'][i].astype('str')+'.png', target_size=(28,28,3), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)
print(test_image.shape)

#way to combine multiple csv file
import os
import glob
import pandas as pd
os.chdir("/mydir")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')

import csv

f = open('change.csv')
inf = csv.reader(f)
for row in inf:
    print(row[1])

with open('change.csv', 'w') as outf:
  writer = csv.writer(outf)
  for row in inf:
    if row[1]==0:
      row[1] = 'airplane'
      writer.writerow(row)
    elif row[1]==1:
      row[1] = 'automobile'
      writer.writerow(row)
    else:
      writer.writerow(row)
    print(row[1])

writer.writerows(inf)
data = pd.read_csv("change.csv")  

dat = [['Austria', 'Germany', 'hob', 'Australia'],
        ['Spain', 'France', 'Italy', 'Mexico']]

df = pd.DataFrame(data, columns = ['A','B'])

# Values to find and their replacements
findL = [0,1,2,3,4,5,6,7,8,9]
replaceL = ['airplane', 'automobile', 'bird', 'cat','deer','dog','frog','horse','ship','truck']

# Select column (can be A,B,C,D)
col = 'B';

# Find and replace values in the selected column
df[col] = df[col].replace(findL, replaceL)
writer.writerow(df)
df.to_csv("change1.csv",index=True)"""
import pandas as pd
from keras.utils import np_utils

"""df = pd.read_csv('change.csv', index_col=False)   #using pandas to read in the CSV file

#let's say in this dataframe you want to do corrections on the 'column for correction' column

correctiondict= {
                  0: 'airplane',
                  1: 'automobile'
                 }

df['B']=df['B'].replace(correctiondict)"""

"""train = pd.read_csv('trainLabels.csv')
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
print(y)
print(type(y))

#convert the labels into categorical  
num_classes=10
y = np_utils.to_categorical(y,num_classes)

ab=pd.read_csv('sample_cnn1.csv')
ab.replace(1,'automobile', inplace=True)
ab.replace(2,'bird', inplace=True)
ab.replace(3,'cat', inplace=True)
ab.replace(4,'deer', inplace=True)
ab.replace(5,'dog', inplace=True)
ab.replace(6,'frog', inplace=True)
ab.replace(7,'horse', inplace=True)
ab.replace(8,'ship', inplace=True)
ab.replace(9,'truck', inplace=True)
ab.replace(0,'airplane', inplace=True)
#ab[ab==0]='airplane'
#ab[ab==1,1]='automobile'
ab.to_csv('cnn.csv',header=True,index=False)

test_data = np.ndarray(((second_index - first_index), 28,28,3),
                 dtype = np.float32)

for i, img_file in enumerate(img_files):
        img = Image.open(img_file)
        test_data[i] = np.array(img)""

first_index = 1; second_index = 50000
img_files= ['test/' + str(i) +  
                    ".png"  for i in range(first_index, (second_index + 1))]

test_data = np.ndarray(((second_index - first_index), 32,32,3),
                 dtype = np.float32)

for i, img_file in enumerate(img_files):
        img = image.load_img(img_file)
        test_data[i] = np.array(img)

"from PIL import Image
import glob
image_list = []
for filename in glob.glob('test/*.png'): #assuming png
    im=image.load_img(filename)
    image_list.append(im)"""

test_image = []
f_index=250001
l_index=300000
for i in tqdm(range(f_index,(l_index+1))):
    img = image.load_img('test/'+str(i)+'.png', target_size=(28,28,3))
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)
print(test_image.shape)


