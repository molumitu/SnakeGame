from ray.rllib.agents.dqn import DQNTrainer
import ray.tune
from env import SnakeGame
from ray.rllib.models.catalog import ModelCatalog
from model import DQN
ModelCatalog.register_custom_model("SnakeModel", DQN)


config = {
    "env": SnakeGame,
    "seed":73,
    "framework":"torch",
    # "model":{
    #     "custom_model": "SnakeModel",
    # },
    "prioritized_replay": True,
    "dueling": ray.tune.grid_search([False, True]),
    "double_q": ray.tune.grid_search([False, True]),
    # "num_atoms": 0,
}

trainer = DQNTrainer(config)
ray.tune.run(
    DQNTrainer,
    config=config,
    stop={
        "episode_reward_mean": 1000,
        "agent_timesteps_total": 600000,
    },
    checkpoint_at_end=True,
    checkpoint_freq=10,
    verbose=1,
    progress_reporter=ray.tune.CLIReporter(print_intermediate_tables=True)
)
