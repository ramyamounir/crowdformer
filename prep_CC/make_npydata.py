import os
import numpy as np

if not os.path.exists('/home/ramy/Documents/crowdformer/data/CraneCounting/November/npydata'):
    os.makedirs('/home/ramy/Documents/crowdformer/data/CraneCounting/November/npydata')


''' for November set'''
try:
    november_train = '/home/ramy/Documents/crowdformer/data/CraneCounting/November/train/images_crop/'
    november_test = '/home/ramy/Documents/crowdformer/data/CraneCounting/November/test/images_crop/'

    train_list = []
    for filename in os.listdir(november_train):
        if filename.split('.')[1] == 'JPG':
            train_list.append(november_train + filename)

    train_list.sort()
    np.save('/home/ramy/Documents/crowdformer/data/CraneCounting/November/npydata/november_train.npy', train_list)

    test_list = []
    for filename in os.listdir(november_test):
        if filename.split('.')[1] == 'JPG':
            test_list.append(november_test + filename)
    test_list.sort()
    np.save('/home/ramy/Documents/crowdformer/data/CraneCounting/November/npydata/november_test.npy', test_list)

    print("generate November image list successfully", len(train_list), len(test_list))
except:
    print("The November dataset path is wrong. Please check you path.")


