import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from GridBased import GridBasedEnv
from LapLoggerCallback import LapLoggerCallback
from stable_baselines3 import DQN, PPO, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from TopDown import TopDownEnv

log_dir_ppo_top = "./logs/ppo_top_logs"
os.makedirs(log_dir_ppo_top, exist_ok=True)

topenv1 = TopDownEnv()
check_env(topenv1)
topenv1 = Monitor(topenv1, log_dir_ppo_top)

topenv1.render()

modelPPO = PPO(
    "MlpPolicy", 
    topenv1, 
    verbose=1, 
    tensorboard_log=log_dir_ppo_top,
    learning_rate=3e-4,
    ent_coef=0.01,
    n_steps=1024,#2048,
    batch_size=64,
    gamma=0.99,
    clip_range=0.2
)

callback_ppo_top = LapLoggerCallback(log_path="logs/ppo_topdown_stats.csv", verbose=1)
modelPPO.learn(total_timesteps=300_000, callback=callback_ppo_top)
modelPPO.save("ppo_topdown_v3")

ppo_top_results = load_results(log_dir_ppo_top)
x, y = ts2xy(ppo_top_results, 'timesteps')
plt.plot(x, y)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("PPO TopDown Learning Curve (with Checkpoints)")
plt.grid(True)
plt.show()

lap_times = []
obs, info = topenv1.reset()
for _ in range(1000):
    action, _ = modelPPO.predict(obs)
    obs, reward, terminated, truncated, info = topenv1.step(action)
    topenv1.render()
    if info.get("lap_time"):
        lap_times.append(info["lap_time"])
    if terminated or truncated:
        obs, info = topenv1.reset()

topenv1.close()

plt.plot(lap_times)
plt.xlabel("Lap")
plt.ylabel("Time (s)")
plt.title("Lap Times - PPO Top-Down Agent")
plt.grid(True)
plt.show()

log_dir_td3_top = "./logs/td3_top_logs"
os.makedirs(log_dir_td3_top, exist_ok=True)

topenv2 = TopDownEnv()
check_env(topenv2)
topenv2 = Monitor(topenv2, log_dir_td3_top)

n_actions = topenv2.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

modelTD3 = TD3(
    "MlpPolicy", 
    topenv2, 
    action_noise=action_noise, 
    verbose=1, 
    tensorboard_log=log_dir_td3_top,
    learning_rate=1e-3,
    batch_size=100,
    buffer_size=100_000,
    train_freq=(1, "episode"),
    gamma=0.98
)

callback_td3_top = LapLoggerCallback(log_path="logs/td3_topdown_stats.csv", verbose=1)
modelTD3.learn(total_timesteps=300_000, callback=callback_td3_top)
modelTD3.save("td3_topdown_v3")

td3_top_results = load_results(log_dir_td3_top)
x, y = ts2xy(td3_top_results, 'timesteps')
plt.plot(x, y)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("TD3 TopDown Learning Curve (with Checkpoints)")
plt.grid(True)
plt.show()

obs, info = topenv2.reset()
lap_times = []
for _ in range(1000):
    action, _ = modelTD3.predict(obs)
    obs, reward, terminated, truncated, _ = topenv2.step(action)
    topenv2.render()
    if info.get("lap_time"):
        lap_times.append(info["lap_time"])
    if terminated or truncated:
        obs, info = topenv2.reset()

topenv2.close()

plt.plot(lap_times)
plt.xlabel("Lap")
plt.ylabel("Time (s)")
plt.title("Lap Times - TD3 Top-Down Agent")
plt.grid(True)
plt.show()

log_dir_ppo_grid = "./logs/ppo_grid_logs/"
os.makedirs(log_dir_ppo_grid, exist_ok=True)

gridenv1 = GridBasedEnv()
check_env(gridenv1)
gridenv1 = Monitor(gridenv1, log_dir_ppo_grid)

#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_gridbased_tensorboard/")
model = PPO(
    "MlpPolicy", 
    gridenv1, 
    verbose=1, 
    tensorboard_log=log_dir_ppo_grid,
    learning_rate=3e-4,
    ent_coef=0.01,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    clip_range=0.2
)
callback_ppo_grid = LapLoggerCallback(log_path="logs/ppo_gridbased_stats.csv", verbose=1)
#model.learn(total_timesteps=100_000)
model.learn(total_timesteps=300_000, callback=callback_ppo_grid)
model.save("ppo_gridbased_v3")

ppo_grid_results = load_results(log_dir_ppo_grid)
x, y = ts2xy(ppo_grid_results, 'timesteps')
plt.plot(x, y)
plt.xlabel('Timesteps')
plt.ylabel('Episode Reward')
plt.title('PPO Grid Learning Curve')
plt.grid(True)
plt.show()

obs, info = gridenv1.reset()
lap_times = []
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = gridenv1.step(action)
    gridenv1.render()
    if info.get("lap_time"):
        lap_times.append(info["lap_time"])
    if terminated or truncated:
        obs, info = gridenv1.reset()

gridenv1.close()

plt.plot(lap_times)
plt.xlabel("Lap")
plt.ylabel("Time (s)")
plt.title("Lap Times - PPO Grid-Based Agent")
plt.grid(True)
plt.show()

log_dir_dqn_grid = "./logs/dqn_grid_logs"
os.makedirs(log_dir_dqn_grid, exist_ok=True)

gridenv2 = GridBasedEnv()
check_env(gridenv2)
gridenv2 = Monitor(gridenv2, log_dir_dqn_grid)

#modelDQN = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_gridbased_tensorboard/")
modelDQN = DQN(
    "MlpPolicy", 
    gridenv2, 
    verbose=1, 
    tensorboard_log=log_dir_dqn_grid,
    learning_rate=1e-3,
    buffer_size=50_000,
    batch_size=64,
    gamma=0.98,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    train_freq=1,
    target_update_interval=500
)
callback_dqn_grid = LapLoggerCallback(log_path="logs/dqn_gridbased_stats.csv", verbose=1)
#model.learn(total_timesteps=100_000)
modelDQN.learn(total_timesteps=300_000, callback=callback_dqn_grid)
modelDQN.save("dqn_gridbased_v3")

dqn_grid_results = load_results(log_dir_dqn_grid)
x, y = ts2xy(dqn_grid_results, 'timesteps')
plt.plot(x, y)
plt.xlabel('Timesteps')
plt.ylabel('Episode Reward')
plt.title('DQN Grid Learning Curve')
plt.grid(True)
plt.show()

obs, info = gridenv2.reset()
lap_times = []
for _ in range(1000):
    action, _ = modelDQN.predict(obs)
    obs, reward, terminated, truncated, info = gridenv2.step(action)
    gridenv2.render()
    if info.get("lap_time"):
        lap_times.append(info["lap_time"])
    if terminated or truncated:
        obs, info = gridenv2.reset()
gridenv2.close()

plt.plot(lap_times)
plt.xlabel("Lap")
plt.ylabel("Time (s)")
plt.title("Lap Times - DQN Grid-Based Agent")
plt.grid(True)
plt.show()

df = pd.read_csv("top_lap_times.csv", names=["Lap", "Time"])
print(df.describe())
df.plot(x="Lap", y="Time", title="Lap Time Trend")

df2 = pd.read_csv("grid_lap_times.csv", names=["Lap", "Time"])
print(df2.describe())
df2.plot(x="Lap", y="Time", title="Lap Time Trend")

#top_env_rec_1 = DummyVecEnv([lambda: TopDownEnv()])
#video_folder = "./videos/"
#video_length = 500

#top_env_rec_1 = VecVideoRecorder(
#    top_env_rec_1,
#    video_folder,
#    record_video_trigger=lambda x: x % 10000 == 0,
#    video_length=video_length,
#    name_prefix="topdown_run"
#)

#model_rec_ppo = PPO(
#    "MlpPolicy",
#    top_env_rec_1,
#    verbose=1,
#    tensorboard_log=log_dir_ppo_top,
#    learning_rate=3e-4,
#    ent_coef=0.01,
#    n_steps=1024,
#    batch_size=64,
#    gamma=0.99,
#    clip_range=0.2
#)
#model_rec_ppo.learn(total_timesteps=30000)
#top_env_rec_1.close()