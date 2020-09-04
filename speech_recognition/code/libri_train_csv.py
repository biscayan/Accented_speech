import csv
import os
import re

###make csv file
libri_csv = open('libri_train.csv','w',newline='',encoding='UTF8')
libri = csv.writer(libri_csv)
libri.writerow(['file','dataset','sentence'])

###regular expression
pattern='[^\w\s\']'
repl=''

###read txt file
txt_dir = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/corpus/Librispeech/train-clean-100/transcript/'
for txt in os.listdir(txt_dir):
    with open(txt_dir+txt,'r',encoding='UTF8') as txt_file:
        lines = txt_file.readlines()
        ###remove new line
        lines = list(map(lambda s: s.strip(), lines))
        for line in lines:
            filename = line.split(' ')[0]
            sentence = ' '.join(line.split(' ')[1:])
            ###write csv file
            libri.writerow([filename+'.wav', 'libri', re.sub(pattern=pattern, repl=repl, string=sentence.upper())])

libri_csv.close()
