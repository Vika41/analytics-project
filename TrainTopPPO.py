import matplotlib.pyplot as plt
import os

from LapLoggerCallback import LapLoggerCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from TopDown import TopDownEnv

log_dir = "./logs/ppo_top_logs"
os.makedirs(log_dir, exist_ok=True)

env = TopDownEnv()
check_env(env)
env = Monitor(env, log_dir)

# For checking where obstacles/checkpoints render, DO NOT USE WHEN TRAINING/EVALUATING
#env.render()

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    ent_coef=0.01,
    #n_steps=2048,
    #batch_size=64,
    gamma=0.99,
    clip_range=0.2
)

callback = LapLoggerCallback(log_path="./logs/ppo_topdown_stats.csv", verbose=1)
model.learn(total_timesteps=300_000, callback=callback)
model.save("ppo_topdown_v8")

results = load_results(log_dir)
x, y = ts2xy(results, 'timesteps')
plt.plot(x, y)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("PPO TopDown Learning Curve (with Checkpoints)")
plt.grid(True)
plt.show()

lap_times = []
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if info.get("lap_time"):
        lap_times.append(info["lap_time"])
    if terminated or truncated:
        obs, info = env.reset()

env.close()

plt.plot(lap_times)
plt.xlabel("Lap")
plt.ylabel("Time (s)")
plt.title("Lap Times - PPO Top-Down Agent")
plt.grid(True)
plt.show()