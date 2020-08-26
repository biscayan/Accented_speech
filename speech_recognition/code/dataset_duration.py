import os
import librosa

def get_duration(file_dir):

    duration = 0

    for files in os.listdir(file_dir):

        file_name = file_dir+'/'+files
        file_len = librosa.get_duration(filename=file_name)
        duration += file_len

    ###convert seconds to hours
    return round(duration/3600)

if __name__=='__main__':
    folder = 'C:/Users/HyeongJu/Desktop/End-to-end_accented/corpus/Common Voice/en/wav dataset/Australia'
    print(get_duration(folder),'h')