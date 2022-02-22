from env import SnakeGame, IllegalMoveError
from trainer import Trainer
from model import DQN
from collections import deque
from utils import *
import matplotlib.pyplot as plt
import torch
import random
import os
import time
from torch.utils.tensorboard import SummaryWriter



class Agent:
    def __init__(self):
        self.epsilon = EPSILON  # randomness
        self.gamma = GAMMA  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.load_model()
        self.trainer = Trainer(self.model)
        # self.trainer.cuda()
        self.env = SnakeGame()
        self.model_epoch = 0

        time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
        self.save_path = f'./results/{time_str}/model'
        self.writer = SummaryWriter()

    def load_model(self):
        try:
            self.model = torch.load(self.save_path + f'/{self.model_epoch}.pth')
            self.model.eval()
        except:
            self.epoch = 0
            self.model = DQN()

    def save_model(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.model_epoch += SAVE
        torch.save(self.model, self.save_path + f'/{self.model_epoch}.pth')

    def render_init(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-0.5, 15.5)
        self.ax.set_ylim(-0.5, 15.5)
        self.ax.plot(0, 0)
        self.fig.canvas.draw()

    def train_long_memory(self):
        if len(self.memory) != 0:
            states, actions, rewards, next_states, dones = zip(*(self.memory))
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        if reward == 0:
            if random.randint(1,10) == 1: self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory.append((state, action, reward, next_state, done))
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):\
        # epsilon-greedy search
        if self.epoch <= 1500*2:
            if random.randint(0,19) == 0:
                return random.randint(0, 3)
        elif self.epoch <= 5000*2:
            if random.randint(0,49) == 0:
                return random.randint(0, 3)
        elif self.epoch <= 15000*2:
            if random.randint(0,299) == 0:
                return random.randint(0, 3)
        state = torch.tensor(state)  # .cuda()
        state = torch.unsqueeze(state, 0)
        prediction = self.model(state)
        action = torch.argmax(prediction).item()
        return action

    def render_update(self):
        snake_head_x = self.env.snake[-1][0]
        snake_head_y = self.env.snake[-1][1]
        point_list_x = [x for (x, y) in self.env.snake]
        point_list_y = [y for (x, y) in self.env.snake]
        plt.cla()
        self.ax.plot(point_list_x, point_list_y, 'r-', snake_head_x, snake_head_y, 'go',
                     self.env.i, self.env.j, 'bo')
        plt.draw()
        self.ax.set_xlim(-0.5, 15.5)
        self.ax.set_ylim(-0.5, 15.5)
        plt.pause(0.03)

    def train(self):
        while True:
            self.env.init()
            if self.epoch % SHOW == 0:
                self.render_init()
            reward = 1
            done = False
            state = self.env.get_state()
            next_state = self.env.get_state()
            while not done:
                state = self.env.get_state()
                action = self.get_action(state)
                try:
                    eaten = self.env.move(action)
                except IllegalMoveError:
                    reward = -20
                    done = True
                except TimeoutError:
                    reward = -25
                    done = True
                else:
                    if self.epoch % SHOW == 0:
                        self.render_update()
                    reward = 40 * int(eaten)
                    next_state = self.env.get_state()
                self.train_short_memory(state, action, reward, next_state, done)
            if self.epoch % SHOW == 0:
                plt.pause(1)
                plt.close()

            self.train_long_memory()
            print(self.epoch, self.env.score, self.env.steps)
            self.writer.add_scalar('Train/Score', self.env.score, self.epoch)
            self.writer.add_scalar('Train/steps', self.env.steps, self.epoch)
            if self.epoch % SAVE == 0:
                self.save_model()
            self.epoch += 1