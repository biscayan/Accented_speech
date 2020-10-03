import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from decoder import GreedyDecoder
from error_rate import WER, CER

###training
def baseline_train(model, device, source_train_loader, target_train_loader, val_loader, criterion, optimizer, epochs, iter_meter, experiment):
    save_path='/home/skgudwn34/Accented_speech/speech_recognition/result/'

    print("Baseline train start")

    model.train()

    with experiment.train():
        for epoch in range(epochs+1):
            for train_idx, (source_train, target_train) in enumerate(zip(source_train_loader,target_train_loader)):

                optimizer.zero_grad()

                ###source data
                source_train_spectrograms, source_train_labels, source_train_input_lengths, source_train_label_lengths = source_train
                source_train_spectrograms, source_train_labels = source_train_spectrograms.to(device), source_train_labels.to(device)
            
                source_train_output = model(source_train_spectrograms)  # (batch, time, n_class)
                source_train_output = F.log_softmax(source_train_output, dim=2)
                source_train_output = source_train_output.transpose(0, 1) # (time, batch, n_class)

                ###target data
                target_train_spectrograms, target_train_labels, target_train_input_lengths, target_train_label_lengths = target_train
                target_train_spectrograms, target_train_labels = target_train_spectrograms.to(device), target_train_labels.to(device)

                target_train_output = model(target_train_spectrograms) # (batch, time, n_class)
                target_train_output = F.log_softmax(target_train_output, dim=2)
                target_train_output = target_train_output.transpose(0, 1) # (time, batch, n_class)
                
                ###loss
                source_train_loss = criterion(source_train_output, source_train_labels, source_train_input_lengths, source_train_label_lengths)
                target_train_loss = criterion(target_train_output, target_train_labels, target_train_input_lengths, target_train_label_lengths)
                
                train_loss = source_train_loss + target_train_loss
                train_loss.backward()

                experiment.log_metric('Train_loss', train_loss.item(), step=iter_meter.get())

                optimizer.step()
                iter_meter.step()
                
            with torch.no_grad():

                val_loss = 0
                val_cer, val_wer = [] , []

                for val_idx, val_data in enumerate(val_loader):

                    val_spectrograms, val_labels, val_input_lengths, val_label_lengths = val_data 
                    val_spectrograms, val_labels = val_spectrograms.to(device), val_labels.to(device)

                    val_output = model(val_spectrograms) # (batch, time, n_class)
                    val_output = F.log_softmax(val_output, dim=2)
                    val_output = val_output.transpose(0, 1) # (time, batch, n_class)

                    loss = criterion(val_output, val_labels, val_input_lengths, val_label_lengths)
                    val_loss += loss.item()/len(val_loader)

                    decoded_preds, decoded_targets = GreedyDecoder(val_output.transpose(0, 1), val_labels, val_label_lengths)

                    for j in range(len(decoded_preds)):
                        val_cer.append(CER(decoded_targets[j], decoded_preds[j]))
                        val_wer.append(WER(decoded_targets[j], decoded_preds[j]))

            avg_cer = sum(val_cer)/len(val_cer)
            avg_wer = sum(val_wer)/len(val_wer)
                
            if epoch % 5 ==0:
                print('Epoch: {:4d}/{} | Train loss: {:.6f}'.format(epoch, epochs, train_loss.item()))
                print('Epoch: {:4d}/{} | Val loss: {:.6f}'.format(epoch, epochs, val_loss))
                print('Average CER: {:.2f}% | Average WER: {:.2f}%\n'.format(round(avg_cer*100,2), round(avg_wer*100,2)))

            #scheduler.step(loss)


###test
def baseline_test(model, device, test_loader, criterion, experiment):
    save_path='/home/skgudwn34/Accented_speech/speech_recognition/result/'
    now = time.localtime()

    print("Baseline test start")
    
    model.eval()
    
    test_loss=0
    test_cer,test_wer=[],[]

    with experiment.test():
        with open(save_path+"%04d_%02d_%02d_%02d_%02d_alignment.txt"%(
                now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min),'w') as align:
            with torch.no_grad():
                for test_idx, test_data in enumerate(test_loader):
                    test_spectrograms, test_labels, test_input_lengths, test_label_lengths = test_data 
                    test_spectrograms, test_labels = test_spectrograms.to(device), test_labels.to(device)

                    test_output = model(test_spectrograms) # (batch, time, n_class)
                    test_output = F.log_softmax(test_output, dim=2)
                    test_output = test_output.transpose(0, 1) # (time, batch, n_class)

                    loss = criterion(test_output, test_labels, test_input_lengths, test_label_lengths)
                    test_loss += loss.item()/len(test_loader)

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