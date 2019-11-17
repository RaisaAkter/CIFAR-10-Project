import os
nb_train=0
nb_test=0
img_size=150
img_channel=3
img_shape=(img_size,img_size,img_channel)

def root_path():
    #return os.path.dirname(__file__)
    return os.path.dirname(os.path.abspath(__file__))

def checkpoint_path():
    return os.path.join(root_path(),"checkpoint")

def dataset_path():
    return os.path.join(root_path(),"dataset")

def src_path():
    return os.path.join(root_path(),"src")

print(src_path())