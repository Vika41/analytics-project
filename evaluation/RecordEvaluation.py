import argparse
import csv
import imageio

from envs.GridBased import GridBasedEnv
from envs.TopDown import TopDownEnv
from evaluation.AnnotateUtils import annotate_frame
from stable_baselines3 import PPO

parser = argparse.ArgumentParser()
parser.add_argument("--env", choices=["topdown", "grid"], required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

env_cls = TopDownEnv if args.env == "topdown" else GridBasedEnv
env = env_cls()
model = PPO.load(args.model)
obs, _ = env.reset()
frames, frame_data = [], []

for step in range(1000):
    frame = env.render(mode='rgb_array')
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, info = env.step(action)

    lap = info.get("current_lap", 0)
    lap_time = info.get("lap_time", None)
    checkpoints = info.get("checkpoints_passed", 0)
    total_checkpoints = getattr(env, "num_checkpoints", 0)
    agent_pos = getattr(env, "agent_pos", None) or getattr(env, "car_pos", None)
    track_size = getattr(env, "grid_size", None) or getattr(env.screen_height, env.screen_width)

    annotated = annotate_frame(frame, action, lap, lap_time, checkpoints, total_checkpoints, agent_pos, track_size)
    frames.append(annotated)

    # CSV buffer
    frame_data.append({
        "frame": len(frames),
        "action": int(action),
        "lap": lap,
        "lap_time": lap_time if lap_time is not None else "",
        "checkpoints": checkpoints,
        "position_x": agent_pos[0] if agent_pos is not None else "",
        "position_y": agent_pos[1] if agent_pos is not None else ""
    })

    if done:
        break

imageio.mimsave(args.out, frames, fps=30)

csv_path = args.out.replace(".mp4", "_metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=frame_data[0].keys())
    writer.writeheader()
    writer.writerows(frame_data)

env.close()
print(f"[VIDEO] Saved to {args.out}")
print(f"[CSV] Saved per-frame metrics to {csv_path}")