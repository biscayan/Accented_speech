import csv
import os
import shutil
import re

###file read
validated_file=open('validated.csv','r',encoding='UTF8')

###regular expression
pattern='[^\w\s\']'
repl=''

###file write
australia_validated=open('australia_validated.csv','w',newline='',encoding='UTF8')
canada_validated=open('canada_validated.csv','w',newline='',encoding='UTF8')
england_validated=open('england_validated.csv','w',newline='',encoding='UTF8')
india_validated=open('india_validated.csv','w',newline='',encoding='UTF8')
us_validated=open('us_validated.csv','w',newline='',encoding='UTF8')

ausv=csv.writer(australia_validated)
canv=csv.writer(canada_validated)
engv=csv.writer(england_validated)
indv=csv.writer(india_validated)
usv=csv.writer(us_validated)

ausv.writerow(['file','accent','sentence'])
canv.writerow(['file','accent','sentence'])
engv.writerow(['file','accent','sentence'])
indv.writerow(['file','accent','sentence'])
usv.writerow(['file','accent','sentence'])

validated_read=csv.reader(validated_file)
validated_header=next(validated_read)

for validated in validated_read:
    if validated[7] == 'australia' :
        ausv.writerow([validated[1].replace('mp3','wav'),validated[7],re.sub(pattern=pattern,repl=repl,string=validated[2].upper())])
    elif validated[7] == 'canada' :
        canv.writerow([validated[1].replace('mp3','wav'),validated[7],re.sub(pattern=pattern,repl=repl,string=validated[2].upper())])
    elif validated[7] == 'england' :
        engv.writerow([validated[1].replace('mp3','wav'),validated[7],re.sub(pattern=pattern,repl=repl,string=validated[2].upper())])
    elif validated[7] == 'indian' :
        indv.writerow([validated[1].replace('mp3','wav'),validated[7],re.sub(pattern=pattern,repl=repl,string=validated[2].upper())])
    elif validated[7] == 'us' :
        usv.writerow([validated[1].replace('mp3','wav'),validated[7],re.sub(pattern=pattern,repl=repl,string=validated[2].upper())])

validated_file.close()

australia_validated.close()
canada_validated.close()
england_validated.close()
india_validated.close()
us_validated.close()
