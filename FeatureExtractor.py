import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, 128))
        self.layers.append(nn.BatchNorm1d(128))
        self.layers.append(nn.Linear(128, self.num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLPDecoder(nn.Module):
    def __init__(self, num_classes, output_dim=3):
        super(MLPDecoder, self).__init__()
        self.num_classes = num_classes
        self.output_dim = output_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.num_classes, 128))
        self.layers.append(nn.BatchNorm1d(128))
        # self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(128, 256))
        self.layers.append(nn.BatchNorm1d(256))
        # self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(256, self.output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#Seq2Seq
class MLPAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(MLPAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        e = self.encoder(x)
        d = self.decoder(e)
        return d