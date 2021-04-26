import torch
import torch.nn as nn


class GRU_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_first=True):
        super(GRU_model,self).__init__()

        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.num_layers=num_layers

        self.gru=nn.GRU(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True) #batch, seq_len, feature

        # Readout layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        out,hn = self.gru(x, h0.detach())

        # Decode the hidden state of the last time step
        out = self.fc(out[:,-1])

        return out