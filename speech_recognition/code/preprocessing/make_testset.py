import csv
import os
import pandas as pd
import torchaudio
import unicodedata

# path수정 필요

# csv path
aus_csv=open('/home/skgudwn34/Accented_speech/speech_recognition/csv/australia_validated.csv','r',encoding='UTF8')
can_csv=open('/home/skgudwn34/Accented_speech/speech_recognition/csv/canada_validated.csv','r',encoding='UTF8')
eng_csv=open('/home/skgudwn34/Accented_speech/speech_recognition/csv/england_validated.csv','r',encoding='UTF8')
ind_csv=open('/home/skgudwn34/Accented_speech/speech_recognition/csv/india_validated.csv','r',encoding='UTF8')
us_csv=open('/home/skgudwn34/Accented_speech/speech_recognition/csv/us_validated.csv','r',encoding='UTF8')

# data path
aus_path='/home/skgudwn34/Accented_speech/speech_recognition/cv4_datasetlim/test_dataset/Australia/'
can_path='/home/skgudwn34/Accented_speech/speech_recognition/cv4_datasetlim/test_dataset/Canada/'
eng_path='/home/skgudwn34/Accented_speech/speech_recognition/cv4_datasetlim/test_dataset/England/'
ind_path='/home/skgudwn34/Accented_speech/speech_recognition/cv4_datasetlim/test_dataset/India/'
us_path='/home/skgudwn34/Accented_speech/speech_recognition/cv4_datasetlim/test_dataset/US/'

save_path='/home/skgudwn34/Accented_speech/speech_recognition/input_data/'

def cleansing(sentence):

    if 'Ø' in sentence:
        sentence = sentence.replace('Ø','O')

    sentence = unicodedata.normalize('NFKD', sentence)
    sentence = sentence.encode('ascii', 'ignore')
    sentence = sentence.decode("utf-8")
    
    return str(sentence)

# make dataset
def make_aus_dataset(data_path,csv_file):

    aus_index=0

    if data_path==aus_path and csv_file==aus_csv:

        file_list=[]

        for filename in os.listdir(data_path):
            file_list.append(filename)
        
        csv_read=csv.reader(csv_file)
        csv_header=next(csv_read)

        df_dict={}

        for csv_line in csv_read:
            if csv_line[0] in file_list:

                waveform, sample_rate=torchaudio.load(data_path+csv_line[0])

                df_dict[aus_index]=(csv_line[0],csv_line[1],cleansing(csv_line[2]),sample_rate,waveform)
                aus_index+=1

        aus_df=pd.DataFrame.from_dict(df_dict, orient='index', columns=['File', 'Accent','Sentence','Sample_rate','Waveform'])

        return aus_df


def make_can_dataset(data_path,csv_file):

    can_index=0

    if data_path==can_path and csv_file==can_csv:

        file_list=[]

        for filename in os.listdir(data_path):
            file_list.append(filename)
        
        csv_read=csv.reader(csv_file)
        csv_header=next(csv_read)

        df_dict={}

        for csv_line in csv_read:
            if csv_line[0] in file_list:

                waveform, sample_rate=torchaudio.load(data_path+csv_line[0])

                df_dict[can_index]=(csv_line[0],csv_line[1],cleansing(csv_line[2]),sample_rate,waveform)
                can_index+=1

        can_df=pd.DataFrame.from_dict(df_dict, orient='index', columns=['File', 'Accent','Sentence','Sample_rate','Waveform'])

        return can_df


def make_eng_dataset(data_path,csv_file):

    eng_index=0

    if data_path==eng_path and csv_file==eng_csv:

        file_list=[]

        for filename in os.listdir(data_path):
            file_list.append(filename)
        
        csv_read=csv.reader(csv_file)
        csv_header=next(csv_read)

        df_dict={}

        for csv_line in csv_read:
            if csv_line[0] in file_list:

                waveform, sample_rate=torchaudio.load(data_path+csv_line[0])

                df_dict[eng_index]=(csv_line[0],csv_line[1],cleansing(csv_line[2]),sample_rate,waveform)
                eng_index+=1

        eng_df=pd.DataFrame.from_dict(df_dict, orient='index', columns=['File', 'Accent','Sentence','Sample_rate','Waveform'])

        return eng_df


def make_ind_dataset(data_path,csv_file):

    ind_index=0

    if data_path==ind_path and csv_file==ind_csv:

        file_list=[]

        for filename in os.listdir(data_path):
            file_list.append(filename)
        
        csv_read=csv.reader(csv_file)
        csv_header=next(csv_read)

        df_dict={}

        for csv_line in csv_read:
            if csv_line[0] in file_list:

                waveform, sample_rate=torchaudio.load(data_path+csv_line[0])

                df_dict[ind_index]=(csv_line[0],csv_line[1],cleansing(csv_line[2]),sample_rate,waveform)
                ind_index+=1

        ind_df=pd.DataFrame.from_dict(df_dict, orient='index', columns=['File', 'Accent','Sentence','Sample_rate','Waveform'])

        return ind_df


def make_us_dataset(data_path,csv_file):

    us_index=0

    if data_path==us_path and csv_file==us_csv:

        file_list=[]

        for filename in os.listdir(data_path):
            file_list.append(filename)
        
        csv_read=csv.reader(csv_file)
        csv_header=next(csv_read)

        df_dict={}

        for csv_line in csv_read:
            if csv_line[0] in file_list:

                waveform, sample_rate=torchaudio.load(data_path+csv_line[0])

                df_dict[us_index]=(csv_line[0],csv_line[1],cleansing(csv_line[2]),sample_rate,waveform)
                us_index+=1

        us_df=pd.DataFrame.from_dict(df_dict, orient='index', columns=['File', 'Accent','Sentence','Sample_rate','Waveform'])

        return us_df

# make dataset
aus_test_df=make_aus_dataset(aus_path, aus_csv)
can_test_df=make_can_dataset(can_path, can_csv)
eng_test_df=make_eng_dataset(eng_path, eng_csv)
ind_test_df=make_ind_dataset(ind_path, ind_csv)
us_test_df=make_us_dataset(us_path, us_csv)

print(aus_test_df)
print(can_test_df)
print(eng_test_df)
print(ind_test_df)
print(us_test_df)

# data save
aus_test_df.to_pickle(save_path+'cv4_aus_testlim')
can_test_df.to_pickle(save_path+'cv4_can_testlim')
eng_test_df.to_pickle(save_path+'cv4_eng_testlim')
ind_test_df.to_pickle(save_path+'cv4_ind_testlim')
us_test_df.to_pickle(save_path+'cv4_us_testlim')

print("save completed")