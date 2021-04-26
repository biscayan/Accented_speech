import torch
from torch.utils.data import Dataset

class Accent_dataset(Dataset): 
  def __init__(self, csv2data,csv2label):

    self.x_data=csv2data[:,:]
    self.y_data=csv2label[:]

    self.x_data = self.x_data.reshape(-1,230, 13)
    self.y_data = self.y_data.reshape(-1)

    self.x_data = torch.cuda.FloatTensor(self.x_data)
    self.y_data = torch.cuda.LongTensor(self.y_data)

    print("=== Dataset Download Complete !!")
    print("X shape:",self.x_data.shape)
    print("Y shape:",self.y_data.shape)

    self.len = len(self.x_data)

  def __getitem__(self, index): 
    return self.x_data[index], self.y_data[index] 

  def __len__(self): 
    return self.len