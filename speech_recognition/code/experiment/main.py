from comet_ml import Experiment
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import os
from cv_dataset import Source_train, Source_val, Source_test, Target_train, Target_val, Target_test
from error_rate import WER, CER
from text_transform import TextTransform
from data_processing import DataProcessing
from asr_model import Baseline, DANN
from baseline_train_test import baseline_train, baseline_test
from DANN_train_test import DANN_train, DANN_test
from iter_meter import IterMeter

def main(learning_rate, batch_size, epochs, experiment):

    # hyperparameters
    hparams = {
        "n_cnn_layers": 4,
        "n_rnn_layers": 4,
        "rnn_dim": 512,
        "n_feats": 128,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    experiment.log_parameters(hparams)
    
    torch.manual_seed(7)

    # device setting
    os.environ["CUDA_VISIBLE_DEVICES"]='1,2'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # dataset
    load_path = load_path = '/home/skgudwn34/Accented_speech/speech_recognition/input_data/'

    source_train_dataset = Source_train(load_path)
    target_train_dataset = Target_train(load_path)

    # source_val_dataset = Source_val(load_path)
    target_val_dataset = Target_val(load_path)

    # source_test_dataset = Source_test(load_path)
    target_test_dataset = Target_test(load_path)

    # train loader
    source_train_loader = data.DataLoader(dataset=source_train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True, drop_last=True,
                                collate_fn=lambda x: DataProcessing(x, 'train'),
                                **kwargs)

    target_train_loader = data.DataLoader(dataset=target_train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True, drop_last=True,
                                collate_fn=lambda x: DataProcessing(x, 'train'),
                                **kwargs)
    # val loader
    # source_val_loader = data.DataLoader(dataset=source_val_dataset,
    #                             batch_size=hparams['batch_size'],
    #                             shuffle=True, drop_last=True,
    #                             collate_fn=lambda x: DataProcessing(x, 'val'),
    #                             **kwargs)

    target_val_loader = data.DataLoader(dataset=target_val_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True, drop_last=True,
                                collate_fn=lambda x: DataProcessing(x, 'val'),
                                **kwargs)
            
    # test loader
    # source_test_loader = data.DataLoader(dataset=source_test_dataset,
    #                             batch_size=hparams['batch_size'],
    #                             shuffle=False, drop_last=True,
    #                             collate_fn=lambda x: DataProcessing(x, 'test'),
    #                             **kwargs)

    target_test_loader = data.DataLoader(dataset=target_test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False, drop_last=True,
                                collate_fn=lambda x: DataProcessing(x, 'test'),
                                **kwargs)

    # model
    model = Baseline(hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'], hparams['n_feats'])
    # model = DANN(hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'], hparams['n_feats'])

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))
    print(model)

    optimizer = optim.Adam(model.parameters(), hparams['learning_rate'])

    label_criterion = nn.CTCLoss(blank=28).to(device)
    domain_criterion = nn.CrossEntropyLoss().to(device)

    iter_meter = IterMeter()
    
    # baseline experiment
    baseline_train(model, device, source_train_loader, target_val_loader, label_criterion, optimizer, epochs, iter_meter, experiment)
    baseline_test(model, device, target_test_loader, label_criterion, experiment)

    # Domain adaptation parameter
    alpha = 0.01

    # DANN experiment
    # DANN_train(model, device, source_train_loader, target_train_loader, target_val_loader, label_criterion, domain_criterion, optimizer, epochs, alpha, iter_meter, experiment)
    # DANN_test(model, device, target_test_loader, label_criterion, experiment)

if __name__=="__main__":
    # comet
    api_key = "c16wdiVuCBod6zyavK5vpgI4i"
    project_name = "accented-speech"
    workspace = "biscayan"
    experiment_name = "cv_ind250_baseline"
    
    experiment = Experiment(api_key=api_key, project_name=project_name, workspace=workspace)
    experiment.set_name(experiment_name)

    # hyperparameters
    learning_rate = 0.0001
    batch_size = 32
    epochs = 100

    main(learning_rate, batch_size, epochs, experiment)