import torch
import matplotlib.pyplot as plt

def Train(gru_model, train_loader, val_loader, optimizer, criterion, device, num_epochs):

    print("Training start")

    train_loss_list, val_loss_list = [], []
    x_step, y_err = [], []

    for epoch in range(num_epochs+1):
        for i, train_data in enumerate(train_loader):

            tain_loss = 0.0

            train_accent, train_label = train_data
        
            train_accent=train_accent.to(device)
            train_label=train_label.to(device)
            
            optimizer.zero_grad()

            hypothesis = gru_model(train_accent).to(device)
            trn_loss = criterion(hypothesis, train_label)

            trn_loss.backward()
            optimizer.step()

            tain_loss+=trn_loss

            with torch.no_grad():

                val_loss=0.0
                correct = 0
                total = 0

                for j,val_data in enumerate(val_loader):
                    val_accent, val_label = val_data
                
                    val_accent=val_accent.to(device)
                    val_label=val_label.to(device)

                    prediction = gru_model(val_accent)
                    vl_loss = criterion(prediction, val_label)
            
                    _, predicted = torch.max(prediction.data, 1)

                    val_loss+=vl_loss*len(val_label)
                    total += val_label.size(0)
                    correct += (predicted == val_label).sum()
                
            correct = int(correct)
            total = int(total)

        if epoch % 10==0:
            print('Epoch: {:4d}/{} | Train loss: {:.6f}'.format(epoch, num_epochs, tain_loss / 100))
            print("Epoch: {:4d}/{} | Validation loss: {:.6f}".format(epoch, num_epochs, val_loss / total))
            print('[Validation set] ERR: %f | ACC: %f (%d / %d)' % (1 - (correct / total), correct / total, correct, total))
            print()
            
        train_loss_list.append(tain_loss/100)
        val_loss_list.append(val_loss/total)
        train_loss = 0.0

        x_step.append(epoch+1)
        y_err.append(1 - (correct / total))

    plt.plot(x_step, train_loss_list, label = "Train Loss")
    plt.plot(x_step, val_loss_list, label = "Validation Loss")
    plt.xlabel('Epochs')
    plt.title('Result_LSTM_LOSS')
    plt.legend()
    plt.savefig('Result_LSTM_LOSS.png')
    plt.show()
    plt.clf()

    plt.plot(x_step, y_err, label = "Validation ERR")
    plt.xlabel('Epochs')
    plt.title('Result_LSTM_ERR')
    plt.legend()
    plt.savefig('Result_LSTM_ERR.png')
    plt.show()
    plt.clf()


def Test(gru_model, test_loader, device):

    print("Test start")

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for k,test_data in enumerate(test_loader):
            test_accent, test_label = test_data

            test_accent=test_accent.to(device)
            test_label=test_label.to(device)

            test_prediction = gru_model(test_accent)

            _, test_predicted = torch.max(test_prediction.data, 1)

            test_total += test_label.size(0)
            test_correct += (test_predicted == test_label).sum()

            test_correct = int(test_correct)
            test_total = int(test_total)

    print('Accuracy of the model on the testset: %d %%' % (100 * test_correct / test_total))