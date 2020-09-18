import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from decoder import GreedyDecoder
from error_rate import WER, CER


###keeps track of total iterations
class IterMeter(object):
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


###training
def Train(model, device, train_loader, val_loader, criterion, optimizer, epochs, iter_meter, experiment):
    save_path='/home/skgudwn34/Accented_speech/speech_recognition/result/'

    print("Train start")

    model.train()

    train_loss_list=[]
    x_step=[]

    with experiment.train():
        for epoch in range(epochs+1):
            for train_idx, train_data in enumerate(train_loader):
                train_spectrograms, train_labels, train_input_lengths, train_label_lengths = train_data
                train_spectrograms, train_labels = train_spectrograms.to(device), train_labels.to(device)

                optimizer.zero_grad()

                train_output = model(train_spectrograms)  # (batch, time, n_class)
                train_output = F.log_softmax(train_output, dim=2)
                train_output = train_output.transpose(0, 1) # (time, batch, n_class)

                train_loss = criterion(train_output, train_labels, train_input_lengths, train_label_lengths)
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

            train_loss_list.append(train_loss)
            x_step.append(epoch+1)

        now = time.localtime()

        plt.plot(x_step, train_loss_list, label = "Train Loss")
        plt.xlabel('Epochs')
        plt.title('LOSS')
        plt.legend()
        plt.savefig(save_path+'%04d_%02d_%02d_%02d_%02d_LOSS.png'%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min))
        plt.show()
        plt.clf()


###test
def Test(model, device, test_loader, criterion, experiment):
    save_path='/home/skgudwn34/Accented_speech/speech_recognition/result/'
    now = time.localtime()

    print("Test start")
    
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