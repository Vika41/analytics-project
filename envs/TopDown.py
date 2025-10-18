import csv
import gymnasium as gym
import numpy as np
import pygame
import time

from gymnasium import spaces

class TopDownEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(TopDownEnv, self).__init__()
        self.screen_width = 800
        self.screen_height = 600
        self.track_color = (255, 255, 255)
        self.car_color = (255, 0, 0)
        self.bg_color = (0, 0, 0)

        self.car_pos = np.array([100.0, 100.0])
        self.car_angle = 0.0
        self.car_speed = 0.0
        self.prev_pos = self.car_pos.copy()
        self.prev_heading = 0.0
        self.car_heading = 0.0
        self.car_velocity = 0.0
        self.max_steering = 1.0
        self.max_throttle = 1.0

        self.checkpoints = []

        self.track_surface = pygame.Surface((self.screen_width, self.screen_height))
        self._build_track()
        self.num_checkpoints = len(self.checkpoints)
        
        obs_dim = 64 * 64 * 3 + self.num_checkpoints + 1
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)
        #self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(64, 64, 3), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.max_steps = 1000
        self.step_count = 0
        self.lap_start_time = None
        self.lap_times = []
        self.max_laps = 3
        self.current_lap = 0

        self.clock = pygame.time.Clock()
        self.done = False

        self.trajectory = np.zeros((self.screen_width, self.screen_height), dtype=np.int32)
        self.passed_checkpoints = set()

    def _build_track(self):
        self.track_surface.fill(self.bg_color)

        # Straights
        pygame.draw.rect(self.track_surface, self.track_color, pygame.Rect(100, 100, 600, 100))     # Top
        pygame.draw.rect(self.track_surface, self.track_color, pygame.Rect(600, 100, 100, 400))     # Right
        pygame.draw.rect(self.track_surface, self.track_color, pygame.Rect(100, 400, 600, 100))     # Bottom
        pygame.draw.rect(self.track_surface, self.track_color, pygame.Rect(100, 200, 100, 200))     # Left

        # Corners
        pygame.draw.arc(self.track_surface, self.track_color, pygame.Rect(100, 100, 100, 100), np.pi, 1.5*np.pi, 10)    # Top-left
        pygame.draw.arc(self.track_surface, self.track_color, pygame.Rect(600, 100, 100, 100), 1.5*np.pi, 0, 10)        # Top-right
        pygame.draw.arc(self.track_surface, self.track_color, pygame.Rect(600, 400, 100, 100), 0, 0.5*np.pi, 10)        # Bottom-right
        pygame.draw.arc(self.track_surface, self.track_color, pygame.Rect(100, 400, 100, 100), 0.5*np.pi, np.pi, 10)    # Bottom-left

        self.finish_line = pygame.Rect(100, 200, 100, 10)
        self.checkpoints = [
            pygame.Rect(400, 100, 10, 100),
            pygame.Rect(600, 300, 100, 10),
            pygame.Rect(300, 400, 10, 100),
            pygame.Rect(100, 350, 100, 10)
        ]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.car_pos = np.array([150.0, 150.0])
        self.car_angle = 0.0
        self.car_speed = 0.0
        self.step_count = 0
        self.done = False
        self.lap_start_time = time.time()
        self.current_lap = 0
        self.lap_times = []
        self.trajectory.fill(0)
        self.passed_checkpoints = set()
        self.prev_heading = 0.0
        self.prev_pos = self.car_pos.copy()

        #x, y = self.car_pos.astype(int)
        #pixel = self.track_surface.get_at((x, y))[:3]
        #print(f"[RESET] Start pixel color at ({x},{y}) = {pixel}")
        print("[RESET] Starting new episode")

        obs = self._get_obs()
        info = {"lap_times": self.lap_times, "current_lap": self.current_lap}
        return obs, info
    
    def step(self, action):
        steer, throttle = np.clip(action, [-1, 0], [1, 1])
        self.car_angle += steer * 5
        self.car_speed = throttle * 2.0

        steer = action[0] * self.max_steering
        throttle = action[1] * self.max_throttle

        self.car_heading += steer
        self.car_velocity += throttle
        self.car_pos += np.array([np.cos(self.car_heading), np.sin(self.car_heading)]) * self.car_velocity

        #dx = self.car_speed * np.cos(np.radians(self.car_angle))
        #dy = self.car_speed * np.sin(np.radians(self.car_angle))
        #self.car_pos += np.array([dx, dy])
        self.step_count += 1

        x, y = self.car_pos.astype(int)
        print(f"[STEP {self.step_count}] pos={self.car_pos}, angle={self.car_angle}, speed={self.car_speed}")
        if 0 <= x < self.screen_width and 0 <= y < self.screen_height:
            pixel = self.track_surface.get_at((x, y))
            #print(f"[PIXEL] color={pixel[:3]} at ({x}, {y})")
            self.trajectory[x, y] += 1

        #reward = 0.5 if self._on_track(self.car_pos) else -1.0
        reward = 0.1
        terminated = False
        truncated = self.step_count >= self.max_steps
        lap_time = None

        if self._on_track(self.car_pos):
            reward += 1.0
            reward += self.car_speed * 0.05
        else:
            reward -= 1.0
            terminated = True
            print(f"[CRASH] Off track at step {self.step_count}, pos={self.car_pos}")

        if self.step_count > 1 and np.linalg.norm(self.car_pos - self.prev_pos) < 1.0:
            reward -= 0.2

        self.prev_pos = self.car_pos.copy()

        remaining = [i for i in range(self.num_checkpoints) if i not in self.passed_checkpoints]
        if remaining:
            print(f"[DEBUG] Remaining checkpoints: {remaining}")
            next_cp = self.checkpoints[remaining[0]]
            cp_center = np.array([next_cp.centerx, next_cp.centery])
            to_cp = cp_center - self.car_pos
            forward_vec = np.array([np.cos(self.car_heading), np.sin(self.car_heading)])
            progress = np.dot(to_cp, forward_vec)
            #dist = np.linalg.norm(cp_center - self.car_pos)
            #reward += max(0, 5.0 - dist / 100.0) * 0.05
            reward += max(0, progress / 100.0) * 0.1

        car = pygame.Rect(self.car_pos[0]-5, self.car_pos[1]-5, 10, 10)
        for i, cp in enumerate(self.checkpoints):
            if i not in self.passed_checkpoints and car.colliderect(cp):
                self.passed_checkpoints.add(i)
                reward += 2.0
                print(f"[CHECKPOINT] Passed checkpoint {i}")

        if self._lap_completed(car):
            lap_time = time.time() - self.lap_start_time
            self._log_lap_time(lap_time)
            self.lap_times.append(lap_time)
            self.lap_start_time = time.time()
            self.current_lap += 1
            reward += max(0, 20.0 - lap_time) # Faster lap = Higher reward
            print(f"[LAP] Completed lap {self.current_lap} in {lap_time:.2f}s")
            self.passed_checkpoints.clear()

            if self.current_lap >= self.max_laps:
                terminated = True
                print("[FINISH] Max laps reached")

        #logger.record("topdown/checkpoints_passed", len(self.passed_checkpoints))
        #logger.record("topdown/lap", self.current_lap)
        #if lap_time:
        #    logger.record("topdown/lap_time")

        info = {
            "lap_times": self.lap_times,
            "current_lap": self.current_lap,
            "lap_time": lap_time if self.current_lap > 0 else None,
            "checkpoints_passed": len(self.passed_checkpoints)
        }
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        checkpoint_obs = np.zeros(self.num_checkpoints, dtype=np.float32)
        for i in self.passed_checkpoints:
            checkpoint_obs[i] = 1.0
        
        obs = np.zeros((64, 64, 3), dtype=np.float32)
        x = int(self.car_pos[0] / (self.screen_width / 64))
        y = int(self.car_pos[1] / (self.screen_height / 64))
        if 0 <= x < 64 and 0 <= y < 64:
            obs[y, x, 0] = 1.0                  # Mark car position
        
        #clamped_speed = max(0.0, min(self.car_speed, 10.0))
        #obs[:, :, 1] = clamped_speed / 10.0    # Normalize speed
        #normalized_angle = self.car_angle % 360
        #obs[:, :, 2] = normalized_angle / 360.0   # Normalize angle
        obs[:, :, 1] = min(max(self.car_speed, 0.0), 10.0) / 10.0
        obs[:, :, 2] = (self.car_angle % 360) / 360.0

        obs = np.concatenate([
            obs.flatten(),
            checkpoint_obs,
            [self.current_lap / self.max_laps]
        ]).astype(np.float32)
        return obs
    
    def _on_track(self, pos):
        x, y = pos.astype(int)
        if 0 <= x < self.screen_width and 0 <= y < self.screen_height:
            pixel = self.track_surface.get_at((x, y))[:3]
            #return pixel[:3] == self.track_color
            return np.linalg.norm(np.array(pixel) - np.array(self.track_color)) < 10
        return False
    
    def _lap_completed(self, car):
        #car_rect = pygame.Rect(self.car_pos[0]-5, self.car_pos[1]-5, 10, 10)
        #return car_rect.colliderect(self.finish_line)
        return len(self.passed_checkpoints) == len(self.checkpoints) and car.colliderect(self.finish_line)
    
    def _log_lap_time(self, lap_time):
        with open("top_lap_times.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.current_lap, lap_time])

    def render(self, mode='human'):
        pygame.init()
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        screen.blit(self.track_surface, (0, 0))
        pygame.draw.circle(screen, self.car_color, self.car_pos.astype(int), 10)

        # Checkpoints
        for i, cp in enumerate(self.checkpoints):
            color = (0, 255, 0) if i in self.passed_checkpoints else(255, 255, 0)
            pygame.draw.rect(screen, color, cp)

        # Finish line
        pygame.draw.rect(screen, (255, 0, 255), self.finish_line)

        # Trajectory heatmap
        max_val = np.max(self.trajectory)
        if max_val > 0:
            for x in range(0, self.screen_width, 4):
                for y in range(0, self.screen_height, 4):
                    count = self.trajectory[x, y]
                    if count > 0:
                        alpha = min(255, int(255 * (count / max_val)))
                        pygame.draw.rect(screen, (0, 255, 0, alpha), pygame.Rect(x, y, 4, 4))
        
        # Lap info
        font = pygame.font.SysFont(None, 24)
        for i, lap_time in enumerate(self.lap_times):
            text = font.render(f"Lap {i+1}: {lap_time:.2f}s", True, (255, 255, 0))
            screen.blit(text, (10, 10 + i * 20))

        cp_text = font.render(f"Checkpoints: {len(self.passed_checkpoints)}", True, (0, 255, 255))
        screen.blit(cp_text, (10, 10 + len(self.lap_times) * 20))

        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(30)
        elif mode == 'rgb_array':
            return pygame.surfarray.array3d(screen).swapaxes(0, 1)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = TopDownEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    env.close()