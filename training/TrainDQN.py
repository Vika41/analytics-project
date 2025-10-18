import matplotlib.pyplot as plt
import os

from GridBased import GridBasedEnv
from LapLoggerCallback import LapLoggerCallback
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

log_dir = "../logs/dqn_grid_logs"
os.makedirs(log_dir, exist_ok=True)

env = GridBasedEnv()
check_env(env)
env = Monitor(env, log_dir)

model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir,
    learning_rate=1e-3,
    buffer_size=50_000,
    batch_size=64,
    gamma=0.98,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    train_freq=1,
    target_update_interval=500
)
callback = LapLoggerCallback(log_path="../logs/dqn_gridbased_stats.csv", verbose=1)
model.learn(total_timesteps=300_000, callback=callback)
model.save("dqn_gridbased_v5")

results = load_results(log_dir)
x, y = ts2xy(results, 'timesteps')
plt.plot(x, y)
plt.xlabel('Timesteps')
plt.ylabel('Episode Reward')
plt.title('DQN Grid Learning Curve')
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
plt.title("Lap Times - DQN Grid-Based Agent")
plt.grid(True)
plt.show()