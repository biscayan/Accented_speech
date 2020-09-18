import os
import shutil
import random

###data split
def aus_split(prefix, train_dir, val_dir, test_dir, data_list):

    random.shuffle(data_list)

    for i in range(0,16000):
        shutil.copy(prefix+'Australia/'+data_list[i], train_dir)

    for j in range(16000,18000):
        shutil.copy(prefix+'Australia/'+data_list[j], val_dir)

    for k in range(18000,20000):
        shutil.copy(prefix+'Australia/'+data_list[k], test_dir)


def can_split(prefix, train_dir, val_dir, test_dir, data_list):

    random.shuffle(data_list)

    for i in range(0,16000):
        shutil.copy(prefix+'Canada/'+data_list[i], train_dir)

    for j in range(16000,18000):
        shutil.copy(prefix+'Canada/'+data_list[j], val_dir)

    for k in range(18000,20000):
        shutil.copy(prefix+'Canada/'+data_list[k], test_dir)


def eng_split(prefix, train_dir, val_dir, test_dir, data_list):

    random.shuffle(data_list)

    for i in range(0,16000):
        shutil.copy(prefix+'England/'+data_list[i], train_dir)

    for j in range(16000,18000):
        shutil.copy(prefix+'England/'+data_list[j], val_dir)

    for k in range(18000,20000):
        shutil.copy(prefix+'England/'+data_list[k], test_dir)


def ind_split(prefix, train_dir, val_dir, test_dir, data_list):

    random.shuffle(data_list)

    for i in range(0,16000):
        shutil.copy(prefix+'India/'+data_list[i], train_dir)

    for j in range(16000,18000):
        shutil.copy(prefix+'India/'+data_list[j], val_dir)

    for k in range(18000,20000):
        shutil.copy(prefix+'India/'+data_list[k], test_dir)


def us_split(prefix, train_dir, val_dir, test_dir, data_list):

    random.shuffle(data_list)

    for i in range(0,16000):
        shutil.copy(prefix+'US/'+data_list[i], train_dir)

    for j in range(16000,18000):
        shutil.copy(prefix+'US/'+data_list[j], val_dir)

    for k in range(18000,20000):
        shutil.copy(prefix+'US/'+data_list[k], test_dir) 


if __name__=='__main__':

    ###file path
    prefix = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/corpus/Common Voice/en/wav dataset/'

    aus_train_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/train_dataset/Australia'
    aus_val_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/val_dataset/Australia'
    aus_test_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/test_dataset/Australia'

    can_train_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/train_dataset/Canada'
    can_val_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/val_dataset/Canada'
    can_test_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/test_dataset/Canada'

    eng_train_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/train_dataset/England'
    eng_val_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/val_dataset/England'
    eng_test_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/test_dataset/England'

    ind_train_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/train_dataset/India'
    ind_val_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/val_dataset/India'
    ind_test_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/test_dataset/India'

    us_train_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/train_dataset/US'
    us_val_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/val_dataset/US'
    us_test_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/dataset/test_dataset/US'

    ###file list
    aus_list = os.listdir(prefix+'Australia') #29521
    can_list = os.listdir(prefix+'Canada') #24358
    eng_list = os.listdir(prefix+'England') #76605
    ind_list = os.listdir(prefix+'India') #31919
    us_list = os.listdir(prefix+'US')

    ###main
    aus_split(prefix, aus_train_dir, aus_val_dir, aus_test_dir, aus_list)
    can_split(prefix, can_train_dir, can_val_dir, can_test_dir, can_list)
    eng_split(prefix, eng_train_dir, eng_val_dir, eng_test_dir, eng_list)
    ind_split(prefix, ind_train_dir, ind_val_dir, ind_test_dir, ind_list)
    us_split(prefix, us_train_dir, us_val_dir, us_test_dir, us_list)


    



