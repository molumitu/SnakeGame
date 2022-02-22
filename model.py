import torch.nn as nn


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(20, 128, bias=True),
            nn.ReLU()
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(128, 16, bias=True),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(16, 4, bias=True)
        )

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            m.weight.data.normal_(0.01, 0.02)
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.hidden(x)
        x = self.hidden1(x)
        x=self.out(x)
        return x