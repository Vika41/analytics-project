import imageio
import numpy as np

from GridBased import GridBasedEnv
from stable_baselines3 import PPO
from TopDown import TopDownEnv

def record_episode(env_class, model_path, output_path, steps=500):
    env = env_class()
    model = PPO.load(model_path)
    obs, _ = env.reset()
    frames = []

    for _ in range(steps):
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, _ , done, _, _ = env.step(action)
        if done:
            break

    #imageio.mimsave(output_path, frames, fps=30)
    writer = imageio.get_writer(output_path, fps=30, codec='libx264', quality=8)
    for frame in frames:
        if frame.shape[2] == 3:
            writer.append_data(frame)
        else:
            writer.append_data(np.squeeze(frame))
    writer.close()
    env.close()
    print(f"[VIDEO] Saved to {output_path}")

record_episode(TopDownEnv, "ppo_topdown_v3.zip", "./videos/ppo_top_eval.mp4")
record_episode(GridBasedEnv, "ppo_gridbased_v3.zip", "./videos/ppo_grid_eval.mp4")