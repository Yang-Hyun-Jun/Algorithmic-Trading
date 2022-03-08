import torch
import torch.nn as nn

class MLP(nn.Module):#피쳐추출벡터.shape[1] + 1

    def __init__(self,
                 input_dim=None, output_dim=None,
                 hidden_act="ReLU", out_act="Identity"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, 16))
        self.layers.append(self.hidden_act)
        self.layers.append(nn.Linear(16, 32))
        self.layers.append(self.hidden_act)
        self.layers.append(nn.Linear(32, 64))
        self.layers.append(self.hidden_act)
        self.layers.append(nn.Linear(64, self.output_dim))
        self.layers.append(self.out_act)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.clip(x, min=-100, max=100)

