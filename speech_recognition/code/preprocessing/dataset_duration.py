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
    train_aus_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/train_dataset/Australia'
    train_can_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/train_dataset/Canada'
    train_eng_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/train_dataset/England'
    train_ind_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/train_dataset/India'
    train_us_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/train_dataset/US'

    val_aus_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/val_dataset/Australia'
    val_can_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/val_dataset/Canada'
    val_eng_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/val_dataset/England'
    val_ind_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/val_dataset/India'
    val_us_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/val_dataset/US'

    test_aus_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/test_dataset/Australia'
    test_can_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/test_dataset/Canada'
    test_eng_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/test_dataset/England'
    test_ind_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/test_dataset/India'
    test_us_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv_datasetALL/test_dataset/US'

    #train_aus = get_duration(train_aus_folder)
    #train_can = get_duration(train_can_folder)
    #train_eng = get_duration(train_eng_folder)
    #train_ind = get_duration(train_ind_folder)
    train_us = get_duration(train_us_folder)
  
    print(train_us)

    #train_duration = train_aus + train_can + train_eng + train_ind + train_us
    #print("train_duration:",train_duration,'h')

    #val_aus = get_duration(val_aus_folder)
    #val_can = get_duration(val_can_folder)
    #val_eng = get_duration(val_eng_folder)
    #val_ind = get_duration(val_ind_folder)
    val_us = get_duration(val_us_folder)

    print(val_us)

    #val_duration = val_aus + val_can + val_eng + val_ind + val_us
    #print("val_duration:",val_duration,'h')

    #test_aus = get_duration(test_aus_folder)
    #test_can = get_duration(test_can_folder)
    #test_eng = get_duration(test_eng_folder)
    #test_ind = get_duration(test_ind_folder)
    test_us = get_duration(test_us_folder)

    print(test_us)

    #test_duration = test_aus + test_can + test_eng + test_ind + test_us
    #print("test_duration:",test_duration,'h')
