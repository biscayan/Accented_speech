import torch
import torchaudio
import torch.nn as nn
from text_transform import TextTransform

# data processing
def DataProcessing(data,data_type):

    # spec augmentation
    # for trainset
    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
        torchaudio.transforms.TimeMasking(time_mask_param=100)
    )

    # for testset
    test_audio_transforms = torchaudio.transforms.MelSpectrogram()

    text_transform = TextTransform()

    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for (_, _, sentence, _, waveform) in data:
        if data_type == 'train':
            spec=train_audio_transforms(waveform).squeeze(0).transpose(0,1)
        elif data_type == 'val':
            spec=test_audio_transforms(waveform).squeeze(0).transpose(0,1)
        elif data_type == 'test':
            spec=test_audio_transforms(waveform).squeeze(0).transpose(0,1)
        else:
            raise Exception('Data_type should be train or test')

        spectrograms.append(spec)

        label = torch.Tensor(text_transform.text_to_int(sentence.upper()))
        labels.append(label)

        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms,batch_first=True).unsqueeze(1).transpose(2,3)
    labels = nn.utils.rnn.pad_sequence(labels,batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths