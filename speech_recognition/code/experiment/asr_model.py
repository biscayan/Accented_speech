import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

###Gradient Reverse Layer
class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x) * alpha

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


###Layer normalization built for cnns input
class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)

        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


###CNN
class feature_extractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(feature_extractor, self).__init__()

        self.cnn = nn.Conv2d(in_channels, out_channels, kernel, stride, padding = kernel//2)
        self.layer_norm = CNNLayerNorm(n_feats)
        self.dropout= nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = self.cnn(x)
        x = self.dropout(x)
        
        return x # (batch, channel, feature, time)


###BiGRU
class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=hidden_size, num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        self.BiGRU.flatten_parameters()
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        
        return x


###baseline
class baseline(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_feats, stride=2, dropout=0.1):
        super(baseline, self).__init__()
        
        n_feats = n_feats//2

        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=1)

        self.cnn_layers=nn.Sequential(*[
            feature_extractor(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) for _ in range(n_cnn_layers)])

        self.fully_connected=nn.Linear(n_feats*32, rnn_dim)

        self.label_predictor=nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2, hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)], nn.Linear(rnn_dim*2, rnn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(rnn_dim, 29))

    def forward(self, x):
        x = self.cnn(x)
        x = self.cnn_layers(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)

        x = self.fully_connected(x)

        x = self.label_predictor(x)

        return x


###DANN
class DANN(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_feats, stride=2, dropout=0.1):
        super(DANN, self).__init__()
        
        n_feats = n_feats//2

        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=1)

        self.cnn_layers=nn.Sequential(*[
            feature_extractor(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) for _ in range(n_cnn_layers)])

        self.fully_connected=nn.Linear(n_feats*32, rnn_dim)
  
        self.label_predictor=nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2, hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)], nn.Linear(rnn_dim*2, rnn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(rnn_dim, 29))

        self.domain_classifier=nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2, hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)], nn.Linear(rnn_dim*2, rnn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(rnn_dim, 2))

    def forward(self, x, alpha):
        x = self.cnn(x)
        x = self.cnn_layers(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)

        x = self.fully_connected(x)

        y = GRL.apply(x, alpha)

        x = self.label_predictor(x)
        y = self.domain_classifier(y)

        return x, y