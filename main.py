from agent import Agent
from utils import TrainParam
if __name__ == '__main__':
    train_param = TrainParam()
    agent = Agent(train_param)
    agent.train()