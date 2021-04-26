import torch
import torch.nn as nn
import torch.optim as optim
from data_manip import csv2data, csv2label
from cv_dataset import Accent_dataset
from model import CNN_model
from torch.utils.data import DataLoader
from train_test import Train, Test


def main(num_epochs, learning_rate, batch_size):

    ### device setting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    ### data manipulation
    train, val, test = csv2data('Australia_mfcc.csv','Canada_mfcc.csv','England_mfcc.csv','India_mfcc.csv','US_mfcc.csv')
    train_label, val_label, test_label = csv2label('Australia_mfcc.csv','Canada_mfcc.csv','England_mfcc.csv','India_mfcc.csv','US_mfcc.csv')

    ### dataset
    train_dataset = Accent_dataset(train, train_label)
    val_dataset = Accent_dataset(val, val_label)  
    test_dataset = Accent_dataset(test, test_label)

    ### data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
    ### model
    cnn_model=CNN_model().to(device)
    print(cnn_model)

    ### loss
    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    ### experiment
    Train(cnn_model, train_loader, val_loader, optimizer, criterion, device, num_epochs)
    Test(cnn_model, test_loader, device)


if __name__ == '__main__':

    ### hyper-parameters
    num_epochs = 150
    learning_rate = 0.00001
    batch_size=256

    main(num_epochs, learning_rate, batch_size)