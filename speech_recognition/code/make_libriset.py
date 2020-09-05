import torchaudio
import os
import csv
import pandas as pd
import unicodedata


###remove diacritic
def remove_accents(sentence):

    sentence = unicodedata.normalize('NFKD', sentence)
    sentence = sentence.encode('ascii', 'ignore')
    sentence = sentence.decode("utf-8")

    return str(sentence)

###make dataset
def make_trainset(data_path,csv_file):

    file_list=[]
    index = 0

    for filename in os.listdir(data_path):
        file_list.append(filename)
        
    csv_read=csv.reader(csv_file)
    csv_header=next(csv_read)

    df_dict={}

    for csv_line in csv_read:
        if csv_line[0] in file_list:

            waveform, sample_rate=torchaudio.load(data_path+csv_line[0])

            df_dict[index]=(csv_line[0],csv_line[1],remove_accents(csv_line[2]),sample_rate,waveform)
            index+=1

    data_df=pd.DataFrame.from_dict(df_dict, orient='index', columns=['File', 'Accent','Sentence','Sample_rate','Waveform'])

    return data_df


def make_valset(data_path,csv_file):

    file_list=[]
    index = 0

    for filename in os.listdir(data_path):
        file_list.append(filename)
        
    csv_read=csv.reader(csv_file)
    csv_header=next(csv_read)

    df_dict={}

    for csv_line in csv_read:
        if csv_line[0] in file_list:

            waveform, sample_rate=torchaudio.load(data_path+csv_line[0])

            df_dict[index]=(csv_line[0],csv_line[1],remove_accents(csv_line[2]),sample_rate,waveform)
            index+=1

    data_df=pd.DataFrame.from_dict(df_dict, orient='index', columns=['File', 'Accent','Sentence','Sample_rate','Waveform'])

    return data_df


def make_testset(data_path,csv_file):

    file_list=[]
    index = 0

    for filename in os.listdir(data_path):
        file_list.append(filename)
        
    csv_read=csv.reader(csv_file)
    csv_header=next(csv_read)

    df_dict={}

    for csv_line in csv_read:
        if csv_line[0] in file_list:

            waveform, sample_rate=torchaudio.load(data_path+csv_line[0])

            df_dict[index]=(csv_line[0],csv_line[1],remove_accents(csv_line[2]),sample_rate,waveform)
            index+=1

    data_df=pd.DataFrame.from_dict(df_dict, orient='index', columns=['File', 'Accent','Sentence','Sample_rate','Waveform'])

    return data_df


if __name__=='__main__':

    #####path수정

    ###csv path
    train_csv=open('/home/skgudwn34/Accented_speech/speech_recognition/csv/libri_train.csv','r',encoding='UTF8')
    val_csv=open('/home/skgudwn34/Accented_speech/speech_recognition/csv/libri_val.csv','r',encoding='UTF8')
    test_csv=open('/home/skgudwn34/Accented_speech/speech_recognition/csv/libri_test.csv','r',encoding='UTF8')


    ###data path
    train_path='/home/skgudwn34/Accented_speech/speech_recognition/libri100h/train_dataset/'
    val_path='/home/skgudwn34/Accented_speech/speech_recognition/libri100h/val_dataset/'
    test_path='/home/skgudwn34/Accented_speech/speech_recognition/libri100h/test_dataset/'


    save_path='/home/skgudwn34/Accented_speech/speech_recognition/input_data/'


    train_df = make_trainset(train_path,train_csv)
    val_df = make_valset(val_path,val_csv)
    test_df = make_testset(test_path,test_csv)

    print(train_df)
    print(val_df)
    print(test_df)

    ###data save
    train_df.to_pickle(save_path+'train_libri')
    val_df.to_pickle(save_path+'val_libri')
    test_df.to_pickle(save_path+'test_libri')
