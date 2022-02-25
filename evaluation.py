from env import SnakeGame
import ray.rllib.agents.dqn as dqn
PATH = R"C:/Users/zgj_t/ray_results/DQNTrainer_2022-02-25_16-32-16/DQNTrainer_RacingEnv_70340_00000_0_double_q=False,dueling=False_2022-02-25_16-32-16/checkpoint_000085/checkpoint-85"
config = {
    "env": SnakeGame,
    "seed":73,
    "framework":"tf2",
    "prioritized_replay": True,
    "dueling": False,
    "double_q": False,
}

trainer = dqn.DQNTrainer(config)
trainer.restore(PATH)

episode_reward = 0
done = False
env = SnakeGame()
obs = env.reset()
while not done:
    action = trainer.compute_action(obs)
    obs, reward, done, info = env.step(action)
    env.render(action)
    episode_reward += reward
print(episode_reward)
