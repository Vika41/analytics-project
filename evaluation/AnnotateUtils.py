import cv2
import numpy as np

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