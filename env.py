import numpy as np
import random
from collections import deque
from utils import EnvParam
import matplotlib.pyplot as plt
import gym
import gym.spaces

class SnakeGame(gym.Env):
    def __init__(self, config=None):
        self.init()
        self.seed()

    def init(self):
        self.snake_board = np.zeros((16, 16), dtype=np.float32)
        self.DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

    def seed(self, seed: int = None):
        self.rng = np.random.default_rng(seed=seed)

    def reset(self):
        self.score = 0
        self.steps = 0
        self.time = 50

        self.snake = deque()
        self.p = random.randint(EnvParam.x_min, EnvParam.x_max)
        self.q = random.randint(EnvParam.y_min, EnvParam.y_max)
        self.snake.append((self.p, self.q))
        self.snake_board[self.p][self.q] = 1

        self.place_food()
        self.eaten = False
        return self.get_obs()

    def place_food(self):
        self.i = self.rng.integers(EnvParam.x_min, EnvParam.x_max)
        self.j = self.rng.integers(EnvParam.y_min, EnvParam.y_max)

    def get_obs(self):
        incoord = lambda m,n: m in range(0,EnvParam.x_max) and n in range(0,EnvParam.y_max)
        grid = lambda m,n: self.snake_board[m+self.p][n+self.q] if incoord(m+self.p, n+self.q) else 0
        return np.array([
            grid(-2, 0),
            grid(-1, -1), grid(-1, 0), grid(-1, 1),
            grid(0, -2), grid(0, -1), grid(0, 0), grid(0, 1), grid(0, 2),
            grid(1, -1), grid(1, 0), grid(1, 1),
            grid(2, 0),
            self.p, self.q, EnvParam.x_max-self.p, EnvParam.y_max-self.q,
            self.p-self.i, self.q-self.j, self.time
        ], dtype=np.float32)

    def step(self, action):
        reward = 1
        done = False
        # move
        self.p, self.q = self.snake[-1]
        self.p += self.DIRECTIONS[action][0]
        self.q += self.DIRECTIONS[action][1]
        self.time -= 1
        if not (0 <= self.p <= EnvParam.x_max and 0 <= self.q <= EnvParam.y_max): # Space limit
            reward = -20
            done = True
        elif self.time < 0: # Time limit
            reward = -20
            done = True
        else:
            self.snake.append((self.p, self.q))
            if not self.eaten: self.snake.popleft()

            # judge food
            self.eaten = False
            food_pos = (self.i, self.j)
            for a in self.snake:
                if a == food_pos:
                    self.eaten = True
                    self.score += 1
                    self.time = 50 + self.score
                    self.place_food()
                    break

            # judge if self collide
            i = 2 if self.eaten else 1
            self.snake_board = np.zeros((EnvParam.x_max+1, EnvParam.y_max+1), dtype=np.float32)
            for (p, q) in self.snake:
                if self.snake_board[p][q] != 0:
                    reward = -20
                    done = True
                self.snake_board[p][q] = i
                i += 1
            reward = 40 * int(self.eaten)
            self.steps += 1
        obs = self.get_obs()

        return obs, reward, done, {"eaten":self.eaten}

    def render_init(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-0.5, 15.5)
        self.ax.set_ylim(-0.5, 15.5)
        self.ax.plot(0, 0)
        self.fig.canvas.draw()

    def render_update(self):
        snake_head_x = self.snake[-1][0]
        snake_head_y = self.snake[-1][1]
        point_list_x = [x for (x, y) in self.snake]
        point_list_y = [y for (x, y) in self.snake]
        plt.cla()
        self.ax.plot(point_list_x, point_list_y, 'r-', snake_head_x, snake_head_y, 'go',
                     self.i, self.j, 'bo')
        plt.draw()
        self.ax.set_xlim(-0.5, 15.5)
        self.ax.set_ylim(-0.5, 15.5)
        plt.pause(0.1)

    def render_complete(self):
        plt.pause(1)
        plt.close()

class IllegalMoveError(BaseException):
    pass

