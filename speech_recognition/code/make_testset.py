import torchaudio
import os
import csv
import pandas as pd
import unicodedata

#####path수정, index수정 필요

###csv path
aus_csv=open('/home/skgudwn34/Accented_speech/speech_recognition/csv/australia_validated.csv','r',encoding='UTF8')
can_csv=open('/home/skgudwn34/Accented_speech/speech_recognition/csv/canada_validated.csv','r',encoding='UTF8')
eng_csv=open('/home/skgudwn34/Accented_speech/speech_recognition/csv/england_validated.csv','r',encoding='UTF8')
ind_csv=open('/home/skgudwn34/Accented_speech/speech_recognition/csv/india_validated.csv','r',encoding='UTF8')
us_csv=open('/home/skgudwn34/Accented_speech/speech_recognition/csv/us_validated.csv','r',encoding='UTF8')

###data path
aus_path='/home/skgudwn34/Accented_speech/speech_recognition/dataset/test_dataset/Australia/'
can_path='/home/skgudwn34/Accented_speech/speech_recognition/dataset/test_dataset/Canada/'
eng_path='/home/skgudwn34/Accented_speech/speech_recognition/dataset/test_dataset/England/'
ind_path='/home/skgudwn34/Accented_speech/speech_recognition/dataset/test_dataset/India/'
us_path='/home/skgudwn34/Accented_speech/speech_recognition/dataset/test_dataset/US/'

save_path='/home/skgudwn34/Accented_speech/speech_recognition/input_data/'

###remove diacritic
def remove_accents(sentence):
    if 'Ø' in sentence:
        sentence = sentence.replace('Ø','O')

    sentence = unicodedata.normalize('NFKD', sentence)
    sentence = sentence.encode('ascii', 'ignore')
    sentence = sentence.decode("utf-8")

    return str(sentence)

###make dataset
def make_dataset(data_path,csv_file):

    aus_index=0
    can_index=1000
    eng_index=2000
    ind_index=3000
    us_index=4000

    ###Australia
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

                df_dict[aus_index]=(csv_line[0],csv_line[1],remove_accents(csv_line[2]),sample_rate,waveform)
                aus_index+=1

        aus_df=pd.DataFrame.from_dict(df_dict, orient='index', columns=['File', 'Accent','Sentence','Sample_rate','Waveform'])

        return aus_df

    ###Canada
    elif data_path==can_path and csv_file==can_csv:

        file_list=[]

        for filename in os.listdir(data_path):
            file_list.append(filename)
        
        csv_read=csv.reader(csv_file)
        csv_header=next(csv_read)

        df_dict={}

        for csv_line in csv_read:
            if csv_line[0] in file_list:

                waveform, sample_rate=torchaudio.load(data_path+csv_line[0])

                df_dict[can_index]=(csv_line[0],csv_line[1],remove_accents(csv_line[2]),sample_rate,waveform)
                can_index+=1

        can_df=pd.DataFrame.from_dict(df_dict, orient='index', columns=['File', 'Accent','Sentence','Sample_rate','Waveform'])

        return can_df

    ###England
    elif data_path==eng_path and csv_file==eng_csv:

        file_list=[]

        for filename in os.listdir(data_path):
            file_list.append(filename)
        
        csv_read=csv.reader(csv_file)
        csv_header=next(csv_read)

        df_dict={}

        for csv_line in csv_read:
            if csv_line[0] in file_list:

                waveform, sample_rate=torchaudio.load(data_path+csv_line[0])

                df_dict[eng_index]=(csv_line[0],csv_line[1],remove_accents(csv_line[2]),sample_rate,waveform)
                eng_index+=1

        eng_df=pd.DataFrame.from_dict(df_dict, orient='index', columns=['File', 'Accent','Sentence','Sample_rate','Waveform'])

        return eng_df

    ###India
    elif data_path==ind_path and csv_file==ind_csv:

        file_list=[]

        for filename in os.listdir(data_path):
            file_list.append(filename)
        
        csv_read=csv.reader(csv_file)
        csv_header=next(csv_read)

        df_dict={}

        for csv_line in csv_read:
            if csv_line[0] in file_list:

                waveform, sample_rate=torchaudio.load(data_path+csv_line[0])

                df_dict[ind_index]=(csv_line[0],csv_line[1],remove_accents(csv_line[2]),sample_rate,waveform)
                ind_index+=1

        ind_df=pd.DataFrame.from_dict(df_dict, orient='index', columns=['File', 'Accent','Sentence','Sample_rate','Waveform'])

        return ind_df

    ###US
    elif data_path==us_path and csv_file==us_csv:


        file_list=[]

        for filename in os.listdir(data_path):
            file_list.append(filename)
        
        csv_read=csv.reader(csv_file)
        csv_header=next(csv_read)

        df_dict={}

        for csv_line in csv_read:
            if csv_line[0] in file_list:

                waveform, sample_rate=torchaudio.load(data_path+csv_line[0])

                df_dict[us_index]=(csv_line[0],csv_line[1],remove_accents(csv_line[2]),sample_rate,waveform)
                us_index+=1

        us_df=pd.DataFrame.from_dict(df_dict, orient='index', columns=['File', 'Accent','Sentence','Sample_rate','Waveform'])

        return us_df


aus_test_df=make_dataset(aus_path,aus_csv)
can_test_df=make_dataset(can_path,can_csv)
eng_test_df=make_dataset(eng_path,eng_csv)
ind_test_df=make_dataset(ind_path,ind_csv)
us_test_df=make_dataset(us_path,us_csv)

test_df=pd.concat([aus_test_df,can_test_df,eng_test_df,ind_test_df,us_test_df])

###data save

test_df.to_pickle(save_path+'test_set') 
#test_df.to_csv(save_path+'test_set.csv',index=False)
