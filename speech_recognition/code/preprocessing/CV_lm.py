import csv
import os
import unicodedata
from csv_cleansing import common

###prefix folder
csv_prefix = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/wav_csv/'
txt_prefix = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/corpus/language_model/'

###open csv
aus_validated = open(csv_prefix+'australia_validated.csv','r',encoding='UTF8')
can_validated = open(csv_prefix+'canada_validated.csv','r',encoding='UTF8')
eng_validated = open(csv_prefix+'england_validated.csv','r',encoding='UTF8')
ind_validated = open(csv_prefix+'india_validated.csv','r',encoding='UTF8')
us_validated = open(csv_prefix+'us_validated.csv','r',encoding='UTF8')

###remove diacritic
def remove_diacritics(sentence):
    if 'Ø' in sentence:
        sentence = sentence.replace('Ø','O')

    sentence = unicodedata.normalize('NFKD', sentence)
    sentence = sentence.encode('ascii', 'ignore')
    sentence = sentence.decode("utf-8")

    return str(sentence)

###csv write
CV_lm = open(txt_prefix+'CV_lm.txt','w',encoding='UTF8')

sentence_list = []

aus_read=csv.reader(aus_validated)
aus_header=next(aus_read)
for aus_data in aus_read:
    aus_sent = remove_diacritics(aus_data[2])
    #aus_sent = common(aus_data[2])
    sentence_list.append(aus_sent)

can_read=csv.reader(can_validated)
can_header=next(can_read)
for can_data in can_read:
    can_sent = remove_diacritics(can_data[2])
    #can_sent = common(can_data[2])
    sentence_list.append(can_sent)

eng_read=csv.reader(eng_validated)
eng_header=next(eng_read)
for eng_data in eng_read:
    eng_sent = remove_diacritics(eng_data[2])
    #eng_sent = common(eng_data[2])
    sentence_list.append(eng_sent)    

ind_read=csv.reader(ind_validated)
ind_header=next(ind_read)
for ind_data in ind_read:
    ind_sent = remove_diacritics(ind_data[2])
    #ind_sent = common(ind_data[2])
    sentence_list.append(ind_sent)

us_read=csv.reader(us_validated)
us_header=next(us_read)
for us_data in us_read:
    us_sent = remove_diacritics(us_data[2])
    #us_sent = common(us_data[2])
    sentence_list.append(us_sent)

sentence_set = sorted(list(set(sentence_list)))

for sent in sentence_set:
    CV_lm.write(sent+'\n')

CV_lm.close()