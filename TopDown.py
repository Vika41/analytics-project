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

        self.car_pos = np.array([150.0, 150.0])

        self.car_angle = 0.0
        self.car_speed = 0.0
        
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(64, 64, 3), dtype=np.float32)

        self.max_steps = 500
        self.step_count = 0
        self.lap_start_time = None
        self.lap_times = []
        self.max_laps = 3
        self.current_lap = 0

        self.track_surface = pygame.Surface((self.screen_width, self.screen_height))
        self._build_track()
        self.clock = pygame.time.Clock()
        self.done = False

        self.trajectory = np.zeros((self.screen_width, self.screen_height), dtype=np.int32)

    def _build_track(self):
        self.track_surface.fill(self.bg_color)
        pygame.draw.rect(self.track_surface, self.track_color, pygame.Rect(100, 100, 600, 400), 20)
        self.finish_line = pygame.Rect(100, 100, 10, 100)

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
        print("[RESET] Starting new episode")
        obs = self._get_obs()
        info = {"lap_times": self.lap_times, "current_lap": self.current_lap}
        return obs, info
    
    def step(self, action):
        steer, throttle = np.clip(action, [-1, 0], [1, 1])
        self.car_angle += steer * 5
        self.car_speed = throttle * 5.0
        dx = self.car_speed * np.cos(np.radians(self.car_angle))
        dy = self.car_speed * np.sin(np.radians(self.car_angle))
        self.car_pos += np.array([dx, dy])
        self.step_count += 1

        #reward = 0.5 if self._on_track(self.car_pos) else -1.0
        reward = 0.1
        terminated = False
        truncated = self.step_count >= self.max_steps
        lap_time = None

        x, y = self.car_pos.astype(int)
        if 0 <= x < self.screen_width and 0 <= y < self.screen_height:
            self.trajectory[x, y] += 1

        print(f"[STEP {self.step_count}] pos={self.car_pos}, angle={self.car_angle}, speed={self.car_speed}")
        print(f"[PIXEL] color={self.track_surface.get_at(self.car_pos.astype(int))}")


        #if self._on_track(self.car_pos):
        #    reward += 1.0
        #    reward += self.car_speed * 0.05
        #else:
        #    reward -= 1.0
        #    terminated = True
        #    print(f"[CRASH] Agent went off track at step {self.step_count}")

        if self._lap_completed():
            lap_time = time.time() - self.lap_start_time
            #self._log_lap_time(lap_time)
            self.lap_times.append(lap_time)
            self.lap_start_time = time.time()
            self.current_lap += 1
            reward += max(0, 20.0 - lap_time) # Faster lap = Higher reward
            print(f"[LAP] Completed lap {self.current_lap} in {lap_time:.2f}s")

            if self.current_lap >= self.max_laps:
                terminated = True
                print("[FINISH] Max laps reached")

        info = {
            "lap_times": self.lap_times,
            "current_lap": self.current_lap,
            "lap_time": lap_time if self.current_lap > 0 else None
        }
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        obs = np.zeros((64, 64, 3), dtype=np.float32)
        x = int(self.car_pos[0] / (self.screen_width / 64))
        y = int(self.car_pos[1] / (self.screen_height / 64))
        if 0 <= x < 64 and 0 <= y < 64:
            obs[y, x, 0] = 1.0                  # Mark car position
        obs[:, :, 1] = self.car_speed / 10.0    # Normalize speed
        obs[:, :, 2] = self.car_angle / 360.0   # Normalize angle
        return obs
    
    def _on_track(self, pos):
        x, y = pos.astype(int)
        if 0 <= x < self.screen_width and 0 <= y < self.screen_height:
            pixel = self.track_surface.get_at((x, y))[:3]
            #return pixel[:3] == self.track_color
            return np.linalg.norm(np.array(pixel) - np.array(self.track_color)) < 10
        return False
    
    def _lap_completed(self):
        car_rect = pygame.Rect(self.car_pos[0]-5, self.car_pos[1]-5, 10, 10)
        return car_rect.colliderect(self.finish_line)
    
    def _log_lap_time(self, lap_time):
        with open("lap_times_topdown.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.current_lap, lap_time])

    def render(self, mode='human'):
        pygame.init()
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        screen.blit(self.track_surface, (0, 0))
        pygame.draw.circle(screen, self.car_color, self.car_pos.astype(int), 10)

        max_val = np.max(self.trajectory)
        if max_val > 0:
            for x in range(0, self.screen_width, 4):
                for y in range(0, self.screen_height, 4):
                    count = self.trajectory[x, y]
                    if count > 0:
                        alpha = min(255, int(255 * (count / max_val)))
                        pygame.draw.rect(screen, (0, 255, 0, alpha), pygame.Rect(x, y, 4, 4))
        
        font = pygame.font.SysFont(None, 24)
        for i, lap_time in enumerate(self.lap_times):
            text = font.render(f"Lap {i+1}: {lap_time:.2f}s", True, (255, 255, 0))
            screen.blit(text, (10, 10 + i * 20))
        
        pygame.display.flip()
        self.clock.tick(30)

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