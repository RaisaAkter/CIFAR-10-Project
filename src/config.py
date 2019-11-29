'''
Author: Raisa
Date: 29-11-19
'''

import os
nb_train=50000
nb_test=300000
img_size=28
img_channel=3
img_shape=(img_size,img_size,img_channel)

lr_1=0.01 
lr_2=0.001
lr_3=0.0001
lr_4=0.00001

def root_path():
    #return os.path.dirname(__file__)
    return os.path.dirname(os.path.abspath(__file__))

def checkpoint_path():
    return os.path.join(root_path(),"checkpoint")

def dataset_path():
    return os.path.join(root_path(),"dataset")

def src_path():
    return os.path.join(root_path(),"src")
