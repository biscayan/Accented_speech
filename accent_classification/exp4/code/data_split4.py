import scipy.io.wavfile as wav
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def data_split(save_path):  

    England_data = np.loadtxt('England_mfcc.csv', delimiter = ",", dtype = np.int32)
    India_data = np.loadtxt('India_mfcc.csv', delimiter = ",", dtype = np.int32)
    US_data = np.loadtxt('US_mfcc.csv', delimiter = ",", dtype = np.int32)

    England_num=2380000
    India_num=2520000
    US_num=2370000

    num=2370000

    divider=237

    train_num=num-(num//5)*2 #1422000
    val_num=num//5 #474000
    test_num=num//5 #474000

    ###data###
    England_train_data = England_data[:train_num,:-1]
    England_val_data = England_data[train_num:train_num+val_num,:-1]
    England_test_data = England_data[train_num+val_num:train_num+val_num+test_num,:-1]

    India_train_data = India_data[:train_num,:-1]
    India_val_data = India_data[train_num:train_num+val_num,:-1]
    India_test_data = India_data[train_num+val_num:train_num+val_num+test_num,:-1]

    US_train_data = US_data[:train_num,:-1] 
    US_val_data = US_data[train_num:train_num+val_num,:-1]
    US_test_data = US_data[train_num+val_num:train_num+val_num+test_num,:-1]

    train_data = np.concatenate((England_train_data,India_train_data,US_train_data),axis=0)
    val_data = np.concatenate((England_val_data,India_val_data,US_val_data),axis=0)
    test_data = np.concatenate((England_test_data,India_test_data,US_test_data),axis=0)

    ###label###
    England_train_label = England_data[:train_num//divider,-1] 
    England_val_label = England_data[train_num:train_num+val_num//divider,-1] 
    England_test_label = England_data[train_num+val_num:train_num+val_num+test_num//divider,-1] 

    India_train_label = India_data[:train_num//divider,-1] 
    India_val_label = India_data[train_num:train_num+val_num//divider,-1] 
    India_test_label = India_data[train_num+val_num:train_num+val_num+test_num//divider,-1] 

    US_train_label = US_data[:train_num//divider,-1] 
    US_val_label = US_data[train_num:train_num+val_num//divider,-1] 
    US_test_label = US_data[train_num+val_num:train_num+val_num+test_num//divider,-1] 

    train_label = np.concatenate((England_train_label,India_train_label,US_train_label),axis=0)
    val_label = np.concatenate((England_val_label,India_val_label,US_val_label),axis=0)
    test_label = np.concatenate((England_test_label,India_test_label,US_test_label),axis=0)

    
    # Training set
    np.save(save_path+"X_train", train_data)
    np.save(save_path+"Y_train", train_label)

    # Validation set
    np.save(save_path+"X_val", val_data)
    np.save(save_path+"Y_val", val_label)

    # Test set
    np.save(save_path+"X_test", test_data)
    np.save(save_path+"Y_test", test_label)
    
save_path='C:/Users/HyeongJu/Desktop/End-to-end_accented/exp/exp4/'

data_split(save_path)