import librosa
import os
import random
import shutil


###data split
def aus_split(prefix, train_dir, val_dir, test_dir):

    aus_list = os.listdir(prefix+'Australia')
    data_list = []

    for i in range(len(aus_list)):

        wave, sr = librosa.load(prefix+'Australia/'+aus_list[i], sr=16000)

        if len(wave) < 150000:
            data_list.append(aus_list[i])    

    print(len(data_list))  
    
    random.shuffle(data_list)

    for i in range(0,25000): 
        shutil.copy(prefix+'Australia/'+data_list[i], train_dir)

    for j in range(25000,27000):
        shutil.copy(prefix+'Australia/'+data_list[j], val_dir)

    for k in range(27000,29000): 
        shutil.copy(prefix+'Australia/'+data_list[k], test_dir)
    

def can_split(prefix, train_dir, val_dir, test_dir):

    can_list = os.listdir(prefix+'Canada')
    data_list = []

    for i in range(len(can_list)):

        wave, sr = librosa.load(prefix+'Canada/'+can_list[i], sr=16000)

        if len(wave) < 150000:
            data_list.append(can_list[i])

    print(len(data_list))
    
    random.shuffle(data_list)

    for i in range(0,4000):
        shutil.copy(prefix+'Canada/'+data_list[i], train_dir)

    for j in range(4000,8000):
        shutil.copy(prefix+'Canada/'+data_list[j], val_dir)

    for k in range(8000,12000):
        shutil.copy(prefix+'Canada/'+data_list[k], test_dir)
    

def eng_split(prefix, train_dir, val_dir, test_dir):

    eng_list = os.listdir(prefix+'England')
    data_list = []

    for i in range(len(eng_list)):

        wave, sr = librosa.load(prefix+'England/'+eng_list[i], sr=16000)

        if len(wave) < 150000:
            data_list.append(eng_list[i])

    print(len(data_list))
    
    random.shuffle(data_list)

    for i in range(0,20000):
        shutil.copy(prefix+'England/'+data_list[i], train_dir)    

    for m in range(20000,22000):
        shutil.copy(prefix+'England/'+data_list[m], val_dir)

    for n in range(22000,24000):
        shutil.copy(prefix+'England/'+data_list[n], test_dir)
    

def ind_split(prefix, train_dir, val_dir, test_dir):

    ind_list = os.listdir(prefix+'India')
    data_list = []

    for i in range(len(ind_list)):

        wave, sr = librosa.load(prefix+'India/'+ind_list[i], sr=16000)

        if len(wave) < 150000:
            data_list.append(ind_list[i])

    print(len(data_list))
    
    random.shuffle(data_list)

    for i in range(0,20000):
        shutil.copy(prefix+'India/'+data_list[i], train_dir)

    for j in range(20000,22000):
        shutil.copy(prefix+'India/'+data_list[j], val_dir)

    for k in range(22000,24000):
        shutil.copy(prefix+'India/'+data_list[k], test_dir)
    

def us_split(prefix, train_dir, val_dir, test_dir):

    us_list = os.listdir(prefix+'US')
    data_list = []

    for i in range(len(us_list)):

        wave, sr = librosa.load(prefix+'US/'+us_list[i], sr=16000)

        if len(wave) < 150000:
            data_list.append(us_list[i])

    print(len(data_list))
    
    random.shuffle(data_list)

    for i in range(0,40000):
        shutil.copy(prefix+'US/'+data_list[i], train_dir)

    for j in range(40000,42000):
        shutil.copy(prefix+'US/'+data_list[j], val_dir)

    for k in range(42000,44000):
        shutil.copy(prefix+'US/'+data_list[k], test_dir) 


if __name__=='__main__':

    ###file path
    prefix = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_validated/'
    
    aus_train_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/train_dataset/Australia'
    aus_val_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/val_dataset/Australia'
    aus_test_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/test_dataset/Australia'
    
    can_train_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/train_dataset/Canada'
    can_val_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/val_dataset/Canada'
    can_test_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/test_dataset/Canada'
    
    eng_train_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/train_dataset/England'
    eng_val_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/val_dataset/England'
    eng_test_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/test_dataset/England'

    ind_train_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/train_dataset/India'
    ind_val_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/val_dataset/India'
    ind_test_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/test_dataset/India'
    
    us_train_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/train_dataset/'
    us_val_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/val_dataset/US'
    us_test_dir = '/home/skgudwn34/Accented_speech/speech_recognition/cv6_1_datasetlim/test_dataset/US'
    

    ###main
    aus_split(prefix, aus_train_dir, aus_val_dir, aus_test_dir)
    can_split(prefix, can_train_dir, can_val_dir, can_test_dir)
    eng_split(prefix, eng_train_dir, eng_val_dir, eng_test_dir)
    ind_split(prefix, ind_train_dir, ind_val_dir, ind_test_dir)
    us_split(prefix, us_train_dir, us_val_dir, us_test_dir)