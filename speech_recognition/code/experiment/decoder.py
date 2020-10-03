import torch
from text_transform import TextTransform

###decoder

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):

    arg_maxes = torch.argmax(output, dim=2)

    decodes = []
    targets = []

    text_transform = TextTransform()

    for i,args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j!=0 and index==args[j-1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))

    return decodes, targets
'''
class BeamCTCDecoder(TextTransform):
    def __init__(self, labels, lm_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100,
                 num_processes=4, blank_index=0):
        super(BeamCTCDecoder, self).__init__(labels)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")

        #labels = labels.replace("'", "a") # TODO fix that
        self._decoder = CTCBeamDecoder(labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, blank_index)
'''