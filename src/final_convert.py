#author Raisa
#Date 17-11-19
#This script is used for convert the 10 class label into its true name and the converted file is kept as a csv file.
#As we predict the test data in 6 iteration, 6 csv file is produced and then we combined them into 1 csv file which is the final csv file for uploed.

import os
import glob
import pandas as pd
import numpy as np

def replace_error(data):
    data.replace(1,'automobile', inplace=True)
    data.replace(2,'bird', inplace=True)
    data.replace(3,'cat', inplace=True)
    data.replace(4,'deer', inplace=True)
    data.replace(5,'dog', inplace=True)
    data.replace(6,'frog', inplace=True)
    data.replace(7,'horse', inplace=True)
    data.replace(8,'ship', inplace=True)
    data.replace(9,'truck', inplace=True)
    data.replace(0,'airplane', inplace=True)


for i in range(0,6):
    data=pd.read_csv('E:/CSE/12th_semester/CIFAR-10-Project/output/sample_cnn_'+str(i)+'.csv')
    if i==1:
        data['id']=data['id']+50000
    elif i==2:
        data['id']=data['id']+100000
    elif i==3:
        data['id']=data['id']+150000
    elif i==4:
        data['id']=data['id']+200000
    elif i==5:
        data['id']=data['id']+250000
    replace_error(data)
    data.to_csv('E:/CSE/12th_semester/CIFAR-10-Project/result/result'+str(i)+'.csv',header=True,index=False)
    

#way to combine multiple csv file

os.chdir("E:/CSE/12th_semester/CIFAR-10-Project/result")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "E:/CSE/12th_semester/CIFAR-10-Project/result/final_csv.csv", index=False, encoding='utf-8-sig')
