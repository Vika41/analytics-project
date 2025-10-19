import matplotlib.pyplot as plt
import os

from GridBased import GridBasedEnv
from LapLoggerCallback import LapLoggerCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

log_dir = "../logs/ppo_grid_logs/"
os.makedirs(log_dir, exist_ok=True)

env = GridBasedEnv()
check_env(env)
env = Monitor(env, log_dir)

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    ent_coef=0.01,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    clip_range=0.2
)
callback = LapLoggerCallback(log_path="../logs/ppo_gridbased_stats.csv", verbose=1)
model.learn(total_timesteps=300_000, callback=callback)
model.save("ppo_gridbased_v5")

ppo_grid_results = load_results(log_dir)
x, y = ts2xy(ppo_grid_results, 'timesteps')
plt.plot(x, y)
plt.xlabel('Timesteps')
plt.ylabel('Episode Reward')
plt.title('PPO Grid Learning Curve')
plt.grid(True)
plt.show()

obs, info = env.reset()
lap_times = []
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
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
plt.title("Lap Times - PPO Grid-Based Agent")
plt.grid(True)
plt.show()