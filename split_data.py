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

#change the lebel in csv
myData = [["first_name", "second_name", "Grade"],
          ['Alex', 'Brian', 'A'],
          ['Tom', 'Smith', 'B']]
 
myFile = open('example2.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(myData)
     
print("Writing complete")
import csv
with open('change.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['label']==0:
            row['label']='airplane'"""
import csv

"""f = open('change.csv')
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

df = pd.read_csv('change.csv', index_col=False)   #using pandas to read in the CSV file

#let's say in this dataframe you want to do corrections on the 'column for correction' column

correctiondict= {
                  0: 'airplane',
                  1: 'automobile'
                 }

df['B']=df['B'].replace(correctiondict)


