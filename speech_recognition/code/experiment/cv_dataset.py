import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple

###commonvoice dataset
class Source_train(Dataset):
    def __init__(self, load_path):

        self.source_train_set = pd.concat([pd.read_pickle(load_path+'cv4_us100_1_trainlim'), 
                                            pd.read_pickle(load_path+'cv4_us100_2_trainlim'),
                                            pd.read_pickle(load_path+'cv4_us25_1_trainlim'),
                                            pd.read_pickle(load_path+'cv4_us25_2_trainlim')])

    def __getitem__(self , idx) -> Tuple[str, str, str, int, Tensor]:
        return (self.source_train_set['File'].iloc[idx], self.source_train_set['Accent'].iloc[idx], self.source_train_set['Sentence'].iloc[idx],
                self.source_train_set['Sample_rate'].iloc[idx], self.source_train_set['Waveform'].iloc[idx])
    
    def __len__(self) -> int:
        return len(self.source_train_set)


class Source_val(Dataset):
    def __init__(self, load_path):

        self.source_val_set = pd.read_pickle(load_path+'cv4_us_vallim')

    def __getitem__(self , idx) -> Tuple[str, str, str, int, Tensor]:
        return (self.source_val_set['File'].iloc[idx], self.source_val_set['Accent'].iloc[idx], self.source_val_set['Sentence'].iloc[idx],
                self.source_val_set['Sample_rate'].iloc[idx], self.source_val_set['Waveform'].iloc[idx])
    
    def __len__(self) -> int:
        return len(self.source_val_set)


class Source_test(Dataset):
    def __init__(self, load_path):

        self.source_test_set = pd.read_pickle(load_path+'cv4_us_testlim')

    def __getitem__(self , idx) -> Tuple[str, str, str, int, Tensor]:
        return (self.source_test_set['File'].iloc[idx], self.source_test_set['Accent'].iloc[idx], self.source_test_set['Sentence'].iloc[idx],
                self.source_test_set['Sample_rate'].iloc[idx], self.source_test_set['Waveform'].iloc[idx])
    
    def __len__(self) -> int:
        return len(self.source_test_set) 


class Target_train(Dataset):
    def __init__(self, load_path):

        self.target_train_set = pd.concat([pd.read_pickle(load_path+'cv_aus_trainlim'), 
                                            pd.read_pickle(load_path+'cv6_1_aus_trainlim')])

    def __getitem__(self , idx) -> Tuple[str, str, str, int, Tensor]:
        return (self.target_train_set['File'].iloc[idx], self.target_train_set['Accent'].iloc[idx], self.target_train_set['Sentence'].iloc[idx],
                self.target_train_set['Sample_rate'].iloc[idx], self.target_train_set['Waveform'].iloc[idx])
    
    def __len__(self) -> int:
        return len(self.target_train_set)


class Target_val(Dataset):
    def __init__(self, load_path):

        self.target_val_set = pd.read_pickle(load_path+'cv_ind_vallim')

    def __getitem__(self , idx) -> Tuple[str, str, str, int, Tensor]:
        return (self.target_val_set['File'].iloc[idx], self.target_val_set['Accent'].iloc[idx], self.target_val_set['Sentence'].iloc[idx],
                self.target_val_set['Sample_rate'].iloc[idx], self.target_val_set['Waveform'].iloc[idx])
    
    def __len__(self) -> int:
        return len(self.target_val_set)
         

class Target_test(Dataset):
    def __init__(self, load_path):

        self.target_test_set = pd.read_pickle(load_path+'cv_ind_testlim')

    def __getitem__(self , idx) -> Tuple[str, str, str, int, Tensor]:
        return (self.target_test_set['File'].iloc[idx], self.target_test_set['Accent'].iloc[idx], self.target_test_set['Sentence'].iloc[idx],
                self.target_test_set['Sample_rate'].iloc[idx], self.target_test_set['Waveform'].iloc[idx])
    
    def __len__(self) -> int:
        return len(self.target_test_set)   