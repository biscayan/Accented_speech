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
def Train(model, device, train_loader, criterion, optimizer, epochs, iter_meter, experiment):
    save_path='/home/skgudwn34/Accented_speech/speech_recognition/result/'

    print("Train start")

    model.train()

    train_loss_list=[]
    x_step=[]

    with experiment.train():
        for epoch in range(epochs+1):
            for batch_idx, data in enumerate(train_loader):
                spectrograms, labels, input_lengths, label_lengths = data
                spectrograms, labels = spectrograms.to(device), labels.to(device)

                optimizer.zero_grad()

                output = model(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # (time, batch, n_class)

                loss = criterion(output, labels, input_lengths, label_lengths)
                loss.backward()

                experiment.log_metric('Train_loss', loss.item(), step=iter_meter.get())

                optimizer.step()
                iter_meter.step()
                
            if epoch % 5 ==0:
                print('Epoch: {:4d}/{} | Train loss: {:.6f}'.format(epoch, epochs, loss.item()))

            train_loss_list.append(loss)
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
                for i, data in enumerate(test_loader):
                    spectrograms, labels, input_lengths, label_lengths = data 
                    spectrograms, labels = spectrograms.to(device), labels.to(device)

                    output = model(spectrograms) # (batch, time, n_class)
                    output = F.log_softmax(output, dim=2)
                    output = output.transpose(0, 1) # (time, batch, n_class)

                    loss = criterion(output, labels, input_lengths, label_lengths)
                    test_loss += loss.item()/len(test_loader)

                    decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)

                    align.write("Index: {}\n".format(i))
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