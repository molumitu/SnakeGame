import torch.nn as nn
from utils import DQNParam

class DQN(nn.Module):
    def __init__(self, dqn_param):
        super().__init__()
        input_size = dqn_param.input_size
        output_size = dqn_param.output_size
        hidden_size = dqn_param.hidden_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size[0], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1], bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size[1], output_size, bias=True)
        )
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            m.weight.data.normal_(0.01, 0.02)
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = self.net(x)
        return x