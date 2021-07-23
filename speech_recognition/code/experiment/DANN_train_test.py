import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import numpy as np
import gc
import torch.nn as nn
from  itertools import zip_longest, cycle
from decoder import GreedyDecoder
from error_rate import WER, CER

# training
def DANN_train(model, device, source_train_loader, target_train_loader, val_loader, label_criterion, domain_criterion, optimizer, epochs, alpha, iter_meter, experiment):
    
    save_path='/home/skgudwn34/Accented_speech/speech_recognition/result/'

    print("DANN train start\n")

    model.train()

    with experiment.train():
        for epoch in range(epochs+1):
            for train_idx, (source_train, target_train) in enumerate(zip_longest(source_train_loader, target_train_loader)):

                optimizer.zero_grad()
                
                # source data
                source_train_spectrograms, source_train_labels, source_train_input_lengths, source_train_label_lengths = source_train
                source_train_spectrograms, source_train_labels = source_train_spectrograms.to(device), source_train_labels.to(device)

                source_train_output, source_domain_output = model(source_train_spectrograms, alpha, mode='label_domain')  # (batch, time, n_class)

                source_train_output = F.log_softmax(source_train_output, dim=2)
                source_train_output = source_train_output.transpose(0, 1) # (time, batch, n_class)

                source_domain_output = source_domain_output.view(source_domain_output.shape[0],source_domain_output.shape[2], source_domain_output.shape[1]).to(device)
                source_domain_label = torch.zeros(source_domain_output.shape[0],source_domain_output.shape[2]).type(torch.LongTensor).to(device)

                # source loss
                source_train_label_loss = label_criterion(source_train_output, source_train_labels, source_train_input_lengths, source_train_label_lengths)
                source_train_domain_loss = domain_criterion(source_domain_output, source_domain_label) #input([B,C]), target([B])

                # target data
                if target_train != None:
                    target_train_spectrograms, target_train_labels, target_train_input_lengths, target_train_label_lengths = target_train
                    target_train_spectrograms, target_train_labels = target_train_spectrograms.to(device), target_train_labels.to(device)

                    target_train_output, target_domain_output = model(target_train_spectrograms, alpha, mode='label_domain') # (batch, time, n_class)
                    
                    target_train_output = F.log_softmax(target_train_output, dim=2)
                    target_train_output = target_train_output.transpose(0, 1) # (time, batch, n_class)

                    target_domain_output = target_domain_output.view(target_domain_output.shape[0],target_domain_output.shape[2], target_domain_output.shape[1]).to(device)
                    target_domain_label = torch.ones(target_domain_output.shape[0],target_domain_output.shape[2]).type(torch.LongTensor).to(device)

                    # target loss
                    target_train_label_loss = label_criterion(target_train_output, target_train_labels, target_train_input_lengths, target_train_label_lengths)
                    target_train_domain_loss = domain_criterion(target_domain_output, target_domain_label)

                    label_loss = source_train_label_loss + target_train_label_loss
                    domain_loss = source_train_domain_loss + target_train_domain_loss
                    trn_loss = label_loss + domain_loss

                elif target_train == None:
                    label_loss = source_train_label_loss
                    domain_loss = source_train_domain_loss
                    trn_loss = label_loss + domain_loss

                # loss
                trn_loss.backward()

                experiment.log_metric('Label_loss', label_loss.item(), step=iter_meter.get())

                optimizer.step()
                iter_meter.step()              

            with torch.no_grad():
                
                val_loss = 0.0
                val_cer, val_wer = [] , []

                for val_idx, val_data in enumerate(val_loader):

                    val_spectrograms, val_labels, val_input_lengths, val_label_lengths = val_data 
                    val_spectrograms, val_labels = val_spectrograms.to(device), val_labels.to(device)

                    val_output= model(val_spectrograms, 0, mode='label_only') # (batch, time, n_class)
                    val_output = F.log_softmax(val_output, dim=2)
                    val_output = val_output.transpose(0, 1) # (time, batch, n_class)

                    vl_loss = label_criterion(val_output, val_labels, val_input_lengths, val_label_lengths)
                    val_loss += vl_loss.item()/len(val_loader)

                    decoded_preds, decoded_targets = GreedyDecoder(val_output.transpose(0, 1), val_labels, val_label_lengths)

                    for j in range(len(decoded_preds)):
                        val_cer.append(CER(decoded_targets[j], decoded_preds[j]))
                        val_wer.append(WER(decoded_targets[j], decoded_preds[j]))

            avg_cer = sum(val_cer)/len(val_cer)
            avg_wer = sum(val_wer)/len(val_wer)
                
            if epoch % 5 ==0:
                print('Epoch: {:4d}/{} | Train loss: {:.6f}'.format(epoch, epochs, trn_loss.item()))
                print('Epoch: {:4d}/{} | Label loss: {:.6f}'.format(epoch, epochs, label_loss.item()))
                print('Epoch: {:4d}/{} | Domain loss: {:.6f}'.format(epoch, epochs, domain_loss.item()))
                print('Epoch: {:4d}/{} | Val loss: {:.6f}'.format(epoch, epochs, val_loss))
                print('Average CER: {:.2f}% | Average WER: {:.2f}%\n'.format(round(avg_cer*100,2), round(avg_wer*100,2)))

# test
def DANN_test(model, device, test_loader, criterion, experiment):
    
    save_path = '/home/skgudwn34/Accented_speech/speech_recognition/result/'
    now = time.localtime()

    print("DANN test start")
    
    model.eval()
    
    test_loss = 0.0
    test_cer, test_wer = [], []

    with experiment.test():
        with open(save_path+"%04d_%02d_%02d_%02d_%02d_alignment.txt"%(
                now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min),'w') as align:
            with torch.no_grad():
                for test_idx, test_data in enumerate(test_loader):
                    test_spectrograms, test_labels, test_input_lengths, test_label_lengths = test_data 
                    test_spectrograms, test_labels = test_spectrograms.to(device), test_labels.to(device)

                    test_output = model(test_spectrograms, 0, mode='label_only') # (batch, time, n_class)
                    test_output = F.log_softmax(test_output, dim=2)
                    test_output = test_output.transpose(0, 1) # (time, batch, n_class)

                    tst_loss = criterion(test_output, test_labels, test_input_lengths, test_label_lengths)
                    test_loss += tst_loss.item()/len(test_loader)

                    decoded_preds, decoded_targets = GreedyDecoder(test_output.transpose(0, 1), test_labels, test_label_lengths)

                    align.write("Index: {}\n".format(test_idx))
                    align.write("Prediction: {}\n".format(decoded_preds))
                    align.write("Reference: {}\n".format(decoded_targets))
                    align.write('\n')

                    for j in range(len(decoded_preds)):
                        test_cer.append(CER(decoded_targets[j], decoded_preds[j]))
                        test_wer.append(WER(decoded_targets[j], decoded_preds[j]))

            avg_cer = sum(test_cer)/len(test_cer)
            avg_wer = sum(test_wer)/len(test_wer)

            print('Test loss: {:.6f} | Average CER: {:.2f}% | Average WER: {:.2f}%\n'.format(
                test_loss, round(avg_cer*100,2), round(avg_wer*100,2)))