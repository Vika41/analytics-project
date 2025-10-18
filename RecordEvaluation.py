import csv
import cv2
import imageio
import numpy as np

from GridBased import GridBasedEnv
from stable_baselines3 import DQN, PPO, TD3
from TopDown import TopDownEnv

def annotate_frame(frame, action, lap, lap_time, checkpoints, total_checkpoints, agent_pos=None, track_size=None):
    """Overlay text onto a frame."""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    #color = (255, 255, 255)
    thickness = 2

    # Text Overlay
    cv2.putText(frame, f"Action: {action}", (10, 25), font, scale, (255, 255, 255), thickness)
    cv2.putText(frame, f"Lap: {lap}", (10, 50), font, scale, (255, 255, 0), thickness)
    if lap_time is not None:
        cv2.putText(frame, f"Lap Time: {lap_time:.2f}s", (10, 75), font, scale, (0, 255, 0), thickness)
    
    cp_color = (0, 255, 0) if total_checkpoints and checkpoints == total_checkpoints else (0, 255, 255)
    cp_text = f"Checkpoints: {checkpoints}/{total_checkpoints}" if total_checkpoints else f"Checkpoints: {checkpoints}"
    cv2.putText(frame, cp_text, (10, 100), font, scale, cp_color, thickness)

    # Minimap
    if agent_pos and track_size:
        minimap = np.zeros((100, 100, 3), dtype=np.uint8)
        tx, ty = track_size
        ax, ay = agent_pos
        mx = int((ay / ty) * 100)
        my = int((ax / tx) * 100)
        cv2.circle(minimap, (mx, my), 4, (0, 0, 255), -1)
        frame[-110:-10, -110:-10] = minimap

    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def record_episode(env_class, model_path, output_path, steps=500):
    frame_data = []
    env = env_class()
    model = PPO.load(model_path)
    obs, _ = env.reset()
    frames = []

    for _ in range(steps):
        frame = env.render(mode='rgb_array')
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)

        lap = info.get("current_lap", 0)
        lap_time = info.get("lap_time", None)
        checkpoints = info.get("checkpoints_passed", 0)
        total_checkpoints = getattr(env, "num_checkpoints", 0)

        # For the minimap
        agent_pos = getattr(env, "agent_pos", None) or getattr(env, "car_pos", None)
        track_size = getattr(env, "grid_size", None) or getattr(env.screen_height, env.screen_width)

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

        annotated = annotate_frame(frame, action, lap, lap_time, checkpoints, total_checkpoints, agent_pos, track_size)
        frames.append(annotated)

        if done:
            break

    #imageio.mimsave(output_path, frames, fps=30)
    writer = imageio.get_writer(output_path, fps=30, codec='libx264', quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    csv_path = output_path.replace(".mp4", "_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=frame_data[0].keys())
        writer.writeheader()
        writer.writerows(frame_data)
    env.close()
    print(f"[VIDEO] Saved to {output_path}")
    print(f"[CSV] Saved per-frame metrics to {csv_path}")

record_episode(TopDownEnv, "ppo_topdown_v4.zip", "./videos/ppo_top_eval.mp4")
record_episode(TopDownEnv, "td3_topdown_v4.zip", "./videos/td3_top_eval.mp4")
record_episode(GridBasedEnv, "ppo_gridbased_v4.zip", "./videos/ppo_grid_eval.mp4")
record_episode(GridBasedEnv, "dqn_gridbased_v4-zip", "./videos/dqn_grid_eval.mp4")