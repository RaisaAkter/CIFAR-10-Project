#author raisa
#DAte 08-11-19
#This script is used for convert the 10 class label into its true name and the converted file is kept as a csv file.
#As we predict the test data in 6 iteration, 6 csv file is produced and then we combined them into 1 csv file which is the final csv file for uploed.

import os
import glob
import pandas as pd
import numpy as np
data=pd.read_csv('sample_cnn1.csv')
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
data.to_csv('result/result1.csv',header=True,index=False)

data=pd.read_csv('sample_cnn2.csv')
data['id']=data['id']+50000
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
data.to_csv('result/result2.csv',header=True,index=False)

data=pd.read_csv('sample_cnn3.csv')
data['id']=data['id']+100000
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
data.to_csv('result/result3.csv',header=True,index=False)

data=pd.read_csv('sample_cnn4.csv')
data['id']=data['id']+150000
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
data.to_csv('result/result4.csv',header=True,index=False)

data=pd.read_csv('sample_cnn5.csv')
data['id']=data['id']+200000
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
data.to_csv('result/result5.csv',header=True,index=False)

data=pd.read_csv('sample_cnn6.csv')
data['id']=data['id']+250000
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
data.to_csv('result/result6.csv',header=True,index=False)

#way to combine multiple csv file

os.chdir("result")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "final_csv.csv", index=False, encoding='utf-8-sig')