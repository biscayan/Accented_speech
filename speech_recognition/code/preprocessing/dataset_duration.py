import librosa
import os

def get_duration(file_dir, accent_type):

    duration = 0

    file_list = os.listdir(file_dir + accent_type)

    for files in file_list:

        file_name = file_dir + accent_type + '/' + files
        file_len = librosa.get_duration(filename=file_name)
        duration += file_len

    # convert seconds to hours
    file_nums = len(file_list)
    dataset_duration =  round(duration/3600)

    print("Accent : ", accent_type)
    print("Number of files : ", file_nums, "개")
    print("Dataset duration : ", dataset_duration, "h")

if __name__=='__main__':

    # Australia, Canada, England, India, US
    train_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv4_datasetlim/train_dataset/'
    val_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv4_datasetlim/val_dataset/'
    test_folder = '/home/skgudwn34/Accented_speech/speech_recognition/cv4_datasetlim/test_dataset/'

    train_duration = get_duration(train_folder, "England")
    val_duration = get_duration(val_folder, "England")
    test_duration = get_duration(test_folder, "England")