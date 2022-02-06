import numpy as np
from numpy.lib.npyio import load
import pandas as pd
import os

data_dir = './data'

def load_data(name = 'coat'):
    if name == 'coat':
        data_set_dir = os.path.join(data_dir,name)
        train_file = os.path.join(data_set_dir,"train.ascii")
        test_file = os.path.join(data_set_dir,'test.ascii')

        with open(train_file,'r') as f:
            x_train = []
            for line in f.readlines():
                x_train.append(line.split())
            
            x_train = np.array(x_train).astype(int)

        with open(test_file,'r') as f:
            x_test = []
            for line in f.readlines():
                x_test.append(line.split())
            
            x_test = np.array(x_test).astype(int)
    


        train,test = rating_mat_to_sample(x_train),rating_mat_to_sample(x_test)
        return train,test
    elif name == "yahoo":
        data_set_dir = os.path.join(data_dir, name)
        train_file = os.path.join(data_set_dir,
            "ydata-ymusic-rating-study-v1_0-train.txt")
        test_file = os.path.join(data_set_dir,
            "ydata-ymusic-rating-study-v1_0-test.txt")

        x_train = []
        # <user_id> <song id> <rating>
        with open(train_file, "r") as f:
            for line in f:
                x_train.append(line.strip().split())
        x_train = np.array(x_train).astype(int)

        x_test = []
        # <user_id> <song id> <rating>
        with open(test_file, "r") as f:
            for line in f:
                x_test.append(line.strip().split())
        x_test = np.array(x_test).astype(int)
        print("===>Load from {} data set<===".format(name))
        print("[train] num data:", x_train.shape[0])
        print("[test]  num data:", x_test.shape[0])

        return x_train,x_test

    else:
        print("Cant find the data set",name)
        return
def rating_mat_to_sample(mat):
    user,item = np.nonzero(mat)
    rates = mat[user,item]
    rates = binaary_rates(rates)
    return np.array([user,item,rates]).T
def used_for_property(mat):
    user,item = np.nonzero(mat)
    rates = mat[user,item]
    rates = binaary_rates(rates)
    return user,item,rates


def binaary_rates(data):
    return np.where(data < 3,0,1)


if __name__ =="__main__":
    x_train,x_test = load_data(name = 'coat')
    print(x_train[0:50])