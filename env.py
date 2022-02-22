import numpy as np
import random
from collections import deque

class SnakeGame():
    def __init__(self):
        self.init()

    def init(self):
        self.score = 0
        self.steps = 0
        self.time = 50
        self.snake_board = np.zeros((16, 16), dtype=np.float32)
        self.snake = deque()
        self.p = random.randint(0, 15)
        self.q = random.randint(0, 15)
        self.snake_board[self.p][self.q] = 1
        self.snake.append((self.p, self.q))
        self.DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
        self.place_food()
        self.eaten = False

    def place_food(self):
        self.i = random.randint(0, 15)
        self.j = random.randint(0, 15)

    def get_state(self):
        incoord = lambda m,n: m in range(0,16) and n in range(0,16)
        grid = lambda m,n: self.snake_board[m+self.p][n+self.q] if incoord(m+self.p, n+self.q) else 0

        return np.array([
            grid(-2, 0),
            grid(-1, -1), grid(-1, 0), grid(-1, 1),
            grid(0, -2), grid(0, -1), grid(0, 0), grid(0, 1), grid(0, 2),
            grid(1, -1), grid(1, 0), grid(1, 1),
            grid(2, 0),
            self.p, self.q, 15-self.p, 15-self.q,
            self.p-self.i, self.q-self.j, self.time
        ], dtype=np.float32)

    def move(self, direction):

        # move
        self.p, self.q = self.snake[-1]
        self.p += self.DIRECTIONS[direction][0]
        self.q += self.DIRECTIONS[direction][1]
        self.time -= 1
        if not (0 <= self.p < 16 and 0 <= self.q < 16): raise IllegalMoveError
        if self.time < 0: raise TimeoutError
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
        self.snake_board = np.zeros((16, 16), dtype=np.float32)
        for (p, q) in self.snake:
            if self.snake_board[p][q] != 0:
                raise IllegalMoveError
            self.snake_board[p][q] = i
            i += 1

        self.steps += 1
        return self.eaten

class IllegalMoveError(BaseException):
    pass