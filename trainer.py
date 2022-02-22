import torch
import torch.optim as optim
import torch.nn as nn
from utils import *

class Trainer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lr = LR
        self.gamma = GAMMA
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            if not done: next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        predict = self.model(state)

        target = predict.clone()

        for i in range(len(done)):
            if done[i]:
                Q_new = reward[i]
            else:
                Q_new = reward[i] + self.gamma * torch.max(self.model(torch.unsqueeze(next_state[i], 0)))

            target[i][action[i]] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, predict)
        loss.backward()
        self.optimizer.step()
