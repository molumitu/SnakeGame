import torch
import torch.optim as optim
import torch.nn as nn
from utils import TrainParam
import copy

class Trainer(nn.Module):
    def __init__(self, model:nn.Module, train_param:TrainParam):
        super().__init__()
        self.lr = train_param.learning_rate
        self.gamma = train_param.gamma
        self.tau = 0.01
        self.model = model
        self.target_model = copy.deepcopy(model)
        for p in self.target_model.parameters():
            p.requires_grad_(False)
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        done = torch.tensor(done).float()

        predict = self.model(state)
        predict = torch.gather(predict,1,action.unsqueeze(0)).squeeze()
        with torch.no_grad():
            target = reward + (1-done) * self.gamma * torch.max(self.target_model(next_state), dim=1).values
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, predict)
        loss.backward()
        self.optimizer.step()
        self.soft_update()
    
    def soft_update(self):
        for source, target in zip(self.model.parameters(), self.target_model.parameters()):
            target *= 1-self.tau
            target += source*self.tau
