from ray.rllib.agents.dqn import DQNTrainer
from env import SnakeGame
config = {
    "env": SnakeGame,
    "seed":73,
    "framework":"torch",
}

trainer = DQNTrainer(config)
while True:
    trainer.train()