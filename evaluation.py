from env import SnakeGame
import ray.rllib.agents.dqn as dqn
PATH = R"C:/Users/zgj_t/ray_results/DQN_2022-02-26_16-34-32/DQN_SnakeGame_ebfd5_00000_0_double_q=False,dueling=False_2022-02-26_16-34-32/checkpoint_000140/checkpoint-140"
config = {
    "env": SnakeGame,
    "seed":5,
    "framework":"torch",
    "prioritized_replay": True,
    "dueling": False,
    "double_q": False,
}

trainer = dqn.DQNTrainer(config)
trainer.restore(PATH)

episode_reward = 0
done = False
env = SnakeGame()
env.render_init()
obs = env.reset()
while not done:
    action = trainer.compute_single_action(obs)
    obs, reward, done, info = env.step(action)
    env.render_update()
    episode_reward += reward
env.render_complete()
print(episode_reward)
