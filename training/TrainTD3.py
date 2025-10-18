import matplotlib.pyplot as plt
import numpy as np
import os

from LapLoggerCallback import LapLoggerCallback
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import load_results, ts2xy
from TopDown import TopDownEnv

log_dir = "../logs/td3_top_logs"
os.makedirs(log_dir, exist_ok=True)

env = TopDownEnv()
check_env(env)
env = Monitor(env, log_dir)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

model = TD3(
    "MlpPolicy", 
    env, 
    action_noise=action_noise, 
    verbose=1, 
    tensorboard_log=log_dir,
    learning_rate=1e-3,
    batch_size=100,
    buffer_size=100_000,
    train_freq=(1, "episode"),
    gamma=0.98
)

callback = LapLoggerCallback(log_path="../logs/td3_topdown_stats.csv", verbose=1)
model.learn(total_timesteps=300_000, callback=callback)
model.save("td3_topdown_v5")

results = load_results(log_dir)
x, y = ts2xy(results, 'timesteps')
plt.plot(x, y)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("TD3 TopDown Learning Curve (with Checkpoints)")
plt.grid(True)
plt.show()

obs, info = env.reset()
lap_times = []
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if info.get("lap_time"):
        lap_times.append(info["lap_time"])
    if terminated or truncated:
        obs, info = env.reset()

env.close()

plt.plot(lap_times)
plt.xlabel("Lap")
plt.ylabel("Time (s)")
plt.title("Lap Times - TD3 Top-Down Agent")
plt.grid(True)
plt.show()