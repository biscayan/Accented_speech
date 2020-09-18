from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple

###commonvoice dataset

class Common_voice(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.len = len(self.dataframe)

    def __getitem__(self , idx) -> Tuple[str, str, str, int, Tensor]:
        return (self.dataframe['File'].iloc[idx], self.dataframe['Accent'].iloc[idx], self.dataframe['Sentence'].iloc[idx],
        self.dataframe['Sample_rate'].iloc[idx], self.dataframe['Waveform'].iloc[idx])
    
    def __len__(self) -> int:
        return self.len