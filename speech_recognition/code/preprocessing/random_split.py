import os
import shutil
import random

###data split
def aus_split(prefix, train_dir, val_dir, test_dir, data_list):

    random.shuffle(data_list)

    for i in range(0,23616):
        shutil.copy(prefix+'Australia/'+data_list[i], train_dir)

    for j in range(23616,26568):
        shutil.copy(prefix+'Australia/'+data_list[j], val_dir)

    for k in range(26568,29520):
        shutil.copy(prefix+'Australia/'+data_list[k], test_dir)


def can_split(prefix, train_dir, val_dir, test_dir, data_list):

    random.shuffle(data_list)

    for i in range(0,19480):
        shutil.copy(prefix+'Canada/'+data_list[i], train_dir)

    for j in range(19480,21915):
        shutil.copy(prefix+'Canada/'+data_list[j], val_dir)

    for k in range(21915,24350):
        shutil.copy(prefix+'Canada/'+data_list[k], test_dir)


def eng_split(prefix, train_dir, val_dir, test_dir, data_list):

    random.shuffle(data_list)

    for i in range(0,61280):
        shutil.copy(prefix+'England/'+data_list[i], train_dir)

    for j in range(61280,68940):
        shutil.copy(prefix+'England/'+data_list[j], val_dir)

    for k in range(68940,76600):
        shutil.copy(prefix+'England/'+data_list[k], test_dir)


def ind_split(prefix, train_dir, val_dir, test_dir, data_list):

    random.shuffle(data_list)

    for i in range(0,25528):
        shutil.copy(prefix+'India/'+data_list[i], train_dir)

    for j in range(25528,28719):
        shutil.copy(prefix+'India/'+data_list[j], val_dir)

    for k in range(28719,31910):
        shutil.copy(prefix+'India/'+data_list[k], test_dir)


def us_split(prefix, train_dir, val_dir, test_dir, data_list):

    random.shuffle(data_list)

    for i in range(0,100000):
        shutil.copy(prefix+'US/'+data_list[i], train_dir)

    for j in range(100000,110000):
        shutil.copy(prefix+'US/'+data_list[j], val_dir)

    for k in range(110000,120000):
        shutil.copy(prefix+'US/'+data_list[k], test_dir) 


if __name__=='__main__':

    ###file path
    prefix = '/home/skgudwn34/Accented_speech/speech_recognition/validated_wav/'

    aus_train_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/train_dataset/Australia'
    aus_val_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/val_dataset/Australia'
    aus_test_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/test_dataset/Australia'

    can_train_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/train_dataset/Canada'
    can_val_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/val_dataset/Canada'
    can_test_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/test_dataset/Canada'

    eng_train_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/train_dataset/England'
    eng_val_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/val_dataset/England'
    eng_test_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/test_dataset/England'

    ind_train_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/train_dataset/India'
    ind_val_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/val_dataset/India'
    ind_test_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/test_dataset/India'

    us_train_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/train_dataset/US'
    us_val_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/val_dataset/US'
    us_test_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/test_dataset/US'

    ###file list
    aus_list = os.listdir(prefix+'Australia')
    can_list = os.listdir(prefix+'Canada') 
    eng_list = os.listdir(prefix+'England') 
    ind_list = os.listdir(prefix+'India') 
    us_list = os.listdir(prefix+'US')

    ###main
    #aus_split(prefix, aus_train_dir, aus_val_dir, aus_test_dir, aus_list)
    #can_split(prefix, can_train_dir, can_val_dir, can_test_dir, can_list)
    #eng_split(prefix, eng_train_dir, eng_val_dir, eng_test_dir, eng_list)
    #ind_split(prefix, ind_train_dir, ind_val_dir, ind_test_dir, ind_list)
    us_split(prefix, us_train_dir, us_val_dir, us_test_dir, us_list)