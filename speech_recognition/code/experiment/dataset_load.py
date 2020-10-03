import pandas as pd
from cv_dataset import Common_voice

###train set
def train_dataset(domain):
    ###data load (pickle file)
    load_path = '/home/skgudwn34/Accented_speech/speech_recognition/input_data/'

    ###source domain
    #source_train = pd.read_pickle(load_path+'cv_us_trainALL')
    source_train = pd.read_pickle(load_path+'libri_train100')

    ###target domain
    target_train = pd.read_pickle(load_path+'cv_aus_trainALL') #Australia
    #target_train = pd.read_pickle(load_path+'cv_can_trainALL') #Canada
    #target_train = pd.read_pickle(load_path+'cv_eng_trainALL') #England
    #target_train = pd.read_pickle(load_path+'cv_ind_trainALL') #India

    if domain == 'source':
        train_set = Common_voice(source_train)
        return train_set

    elif domain == 'target':
        train_set = Common_voice(target_train)
        return train_set


###val set
def val_dataset(domain):
    ###data load (pickle file)
    load_path = '/home/skgudwn34/Accented_speech/speech_recognition/input_data/'

    ###source domain
    #source_val = pd.read_pickle(load_path+'cv_us_valALL')
    source_val = pd.read_pickle(load_path+'libri_val')

    ###target domain
    target_val = pd.read_pickle(load_path+'cv_aus_valALL') #Australia
    #target_val = pd.read_pickle(load_path+'cv_can_valALL') #Canada
    #target_val = pd.read_pickle(load_path+'cv_eng_valALL') #England
    #target_val = pd.read_pickle(load_path+'cv_ind_valALL') #India

    if domain == 'source':
        val_set = Common_voice(source_val)
        return val_set

    elif domain == 'target':
        val_set = Common_voice(target_val)
        return val_set


###test set
def test_dataset(domain):
    ###data load (pickle file)
    load_path = '/home/skgudwn34/Accented_speech/speech_recognition/input_data/'

    ###source domain
    #source_test = pd.read_pickle(load_path+'cv_us_testALL')
    source_test = pd.read_pickle(load_path+'libri_test')

    ###target domain
    target_test = pd.read_pickle(load_path+'cv_aus_testALL') #Australia
    #target_test = pd.read_pickle(load_path+'cv_can_testALL') #Canada
    #target_test = pd.read_pickle(load_path+'cv_eng_testALL') #England
    #target_test = pd.read_pickle(load_path+'cv_ind_testALL') #India

    if domain == 'source':
        test_set = Common_voice(source_test)
        return test_set

    elif domain == 'target':
        test_set = Common_voice(target_test)
        return test_set