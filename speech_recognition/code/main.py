from comet_ml import Experiment
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import pandas as pd

from cv_dataset import Common_voice
from error_rate import WER, CER
from text_transform import TextTransform
from data_processing import DataProcessing
from asr_model import SpeechRecognitionModel
from train_test import IterMeter, Train, Test


def main(learning_rate, batch_size, epochs, experiment):

    ###hyper parameters
    hparams = {
        "n_cnn_layers": 1,
        "n_rnn_layers": 1,
        "rnn_dim": 256,
        "n_class": 29,
        "n_feats": 128,
        "stride":2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    experiment.log_parameters(hparams)

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    ###data load (pickle file)
    load_path = '/home/skgudwn34/Accented_speech/speech_recognition/input_data/'

    train_set = load_path+'train_set'
    train_df = pd.read_pickle(train_set)

    val_set = load_path+'val_set'
    val_df = pd.read_pickle(val_set)

    test_set = load_path+'test_set'
    test_df = pd.read_pickle(test_set)

    train_dataset = Common_voice(train_df)
    val_dataset = Common_voice(val_df)
    test_dataset = Common_voice(test_df)

    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True, drop_last=True,
                                collate_fn=lambda x: DataProcessing(x, 'train'),
                                **kwargs)

    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False, drop_last=True,
                                collate_fn=lambda x: DataProcessing(x, 'test'),
                                **kwargs)

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        )

    ###multi-gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)

    iter_meter = IterMeter()

    Train(model, device, train_loader, criterion, optimizer, epochs, iter_meter, experiment)
    Test(model, device, test_loader, criterion, experiment)


if __name__=="__main__":
    ###comet
    api_key = "c16wdiVuCBod6zyavK5vpgI4i"
    project_name = "accented-speech"
    workspace = "biscayan"
    experiment_name = "accented_speech4"
    
    experiment = Experiment(api_key=api_key, project_name=project_name, workspace=workspace)
    experiment.set_name(experiment_name)

    ###parameters
    learning_rate = 0.0001
    batch_size = 64
    epochs = 100

    main(learning_rate, batch_size, epochs, experiment)