from env import SnakeGame, IllegalMoveError
from trainer import Trainer
from model import DQN
from collections import deque
from utils import EnvParam, TrainParam, DQNParam
import torch
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import TrainParam


class Agent:
    def __init__(self, train_param:TrainParam):
        self.epsilon = train_param.epsilon
        self.buffer_size = train_param.buffer_size
        self.warmup_size = train_param.warmup_size
        self.buffer = deque(maxlen=self.buffer_size)  # popleft()
        self.save_interval = train_param.save_interval
        self.evaluate_interval = train_param.evaluate_interval
        self.batch_size = train_param.batch_size
        self.rng = np.random.default_rng(seed= EnvParam.seed)
        time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
        self.save_path = f'./results/{time_str}/model'

        self.model_epoch = 0
        self.load_model()

        self.trainer = Trainer(self.model, train_param)
        self.env = SnakeGame()
        self.env.reset()
        self.writer = SummaryWriter()

    def load_model(self):
        try:
            self.model = torch.load(self.save_path + f'/{self.model_epoch}.pth')
            self.model.eval()
        except:
            self.epoch = 0
            dqn_param = DQNParam()
            self.model = DQN(dqn_param)

    def save_model(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.model_epoch += self.save_interval
        torch.save(self.model, self.save_path + f'/{self.model_epoch}.pth')

    def train_from_buffer(self):
        if len(self.buffer) != 0:
            states, actions, rewards, next_states, dones = zip(*(self.buffer))
            # random_index = self.rng.integers(low=0, high=len(self.buffer), size=self.batch_size)
            random_index = [i for i in range(len(self.buffer))]
            random_states = []
            random_actions = []
            random_rewards = []
            random_next_states = []
            random_dones = []
            for i in random_index:
                random_states.append(states[i])
                random_actions.append(actions[i])
                random_rewards.append(rewards[i])
                random_next_states.append(next_states[i])
                random_dones.append(dones[i])
            self.trainer.train_step(np.array(random_states), np.array(random_actions), np.array(random_rewards), np.array(random_next_states), np.array(random_dones))

    def get_action(self, state):\
        # epsilon-greedy search
        if self.epoch <= 15000:
            if self.rng.integers(0,20) == 0:
                return self.rng.integers(0,4)
        elif self.epoch <= 50000:
            if self.rng.integers(0,50) == 0:
                return self.rng.integers(0,4)
        elif self.epoch <= 150000:
            if self.rng.integers(0,300) == 0:
                return self.rng.integers(0,4)
        state = torch.tensor(state)
        state = torch.unsqueeze(state, 0)
        prediction = self.model(state)
        action = torch.argmax(prediction).item()
        return action

    def train(self):
        while True:
            self.env.reset()
            # if self.epoch % self.evaluate_interval == 0:
            #     self.evaluate()
            # if self.epoch % self.save_interval == 0:
            #     self.save_model()
            reward = 1
            done = False
            while not done:
                state = self.env.get_obs()
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.buffer.append((state, action, reward, next_state, done))
                if len(self.buffer) > self.warmup_size:
                    self.train_from_buffer()
            self.writer.add_scalar('Train/Score', self.env.score, self.epoch)
            self.writer.add_scalar('Train/Steps', self.env.steps, self.epoch)
            self.epoch += 1
