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
        self.debug = False

        self.car_pos = np.array([150.0, 150.0])
        self.car_angle = 0.0
        self.car_speed = 0.0
        self.car_heading = 0.0
        self.car_velocity = 0.0
        self.car_radius = 10

        self.prev_heading = 0.0
        self.prev_pos = self.car_pos.copy()

        self.max_steering = 0.3
        self.max_throttle = 1.0
        self.max_velocity = 5.0

        self.num_obstacles = 5
        self.obstacle_radius = 10
        self.obstacles = []

        self.checkpoints = []

        self.track_outline = [
            (100, 100), (600, 100), (700, 100), (700, 500),
            (100, 500), (100, 100), (100, 200), (100, 200)
        ]
        #self.track_outline = []
        self.track_mask = pygame.Surface((self.screen_width, self.screen_height))
        self.track_mask.fill((0, 0, 0))
        self.track_width = 40
        self.track_surface = pygame.Surface((self.screen_width, self.screen_height))
        self._build_track()
        self.num_checkpoints = len(self.checkpoints)
        
        obs_dim = 64 * 64 * 3 + self.num_checkpoints + 1
        #self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        #self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(64, 64, 3), dtype=np.float32)
        #self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

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
            pygame.Rect(450, 400, 10, 100),
            pygame.Rect(100, 350, 100, 10)
        ]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.obstacles = []

        #cp_centers = [np.array([cp.centerx, cp.centery]) for cp in self.checkpoints]
        #left_edge, right_edge = [], []

        #for i in range(len(cp_centers)):
        #    p1 = cp_centers[i]
        #    p2 = cp_centers[(i + 1) % len(cp_centers)]
        #    direction = p2 - p1
        #    norm = direction / (np.linalg.norm(direction) + 1e-6)
        #    perp = np.array([-norm[1], norm[0]])

        #    left = p1 + perp * self.track_width
        #    right = p1 - perp * self.track_width
        #    left_edge.append(left)
        #    right_edge.append(right)

        #self.track_outline = [pt.astype(int) for pt in left_edge + right_edge[::-1]]
        self.track_mask = pygame.Surface((self.screen_width, self.screen_height))
        self.track_mask.fill((0, 0, 0))
        pygame.draw.polygon(self.track_mask, (255, 255, 255), self.track_outline)

        self.car_pos = np.array([150.0, 150.0])

        for _ in range(self.num_obstacles):
            for _ in range(100):
                x = np.random.randint(50, self.screen_width - 50)
                y = np.random.randint(50, self.screen_height - 50)
                pos = np.array([x, y])

                if np.linalg.norm(pos - self.car_pos) < 100:
                    continue
                if any(np.linalg.norm(pos - np.array([cp.centerx, cp.centery])) < 100 for cp in self.checkpoints):
                    continue
                if self.finish_line.collidepoint(*pos):
                    continue
                if self.track_mask.get_at((x, y)) != (255, 255, 255, 255):
                    continue

                #if hasattr(self, "track_polygon") and not self.track_polygon.collidepoint:
                #    continue

                self.obstacles.append(pos)
                break

        #self.car_pos = np.array([start_x, start_y], dtype=np.float32)
        self.car_angle = 0.0
        self.car_speed = 0.0
        self.car_velocity = 0.0
        self.car_heading = 0.0

        self.prev_heading = 0.0
        self.prev_pos = self.car_pos.copy()
        self.passed_checkpoints = set()

        self.step_count = 0
        self.done = False

        self.current_lap = 0
        self.lap_times = []
        self.trajectory.fill(0)
        self.lap_start_time = time.time()

        #x, y = self.car_pos.astype(int)
        #pixel = self.track_surface.get_at((x, y))[:3]
        #print(f"[RESET] Start pixel color at ({x},{y}) = {pixel}")
        print("[RESET] Starting new episode")

        obs = self._get_obs()
        info = {"lap_times": self.lap_times, "current_lap": self.current_lap}
        return obs, info
    
    def step(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0)) * self.max_steering
        throttle = float(np.clip(action[1], -1.0, 1.0)) * self.max_throttle

        self.car_heading += steer
        #self.car_heading = (self.car_heading + steer) % (2 * np.pi)
        self.car_velocity = np.clip(self.car_velocity + throttle, 0.0, self.max_velocity)
        self.car_pos += np.array([np.cos(self.car_heading), np.sin(self.car_heading)]) * self.car_velocity

        #steer = np.clip(steer, -self.max_steering, self.max_steering)

        self.step_count += 1
        reward = 0.0
        terminated = False
        info = {}

        for ob in self.obstacles:
            if np.linalg.norm(self.car_pos - ob) < self.obstacle_radius + self.car_radius:
                reward -= 1.0
                terminated = True
                break

        remaining = [i for i in range(self.num_checkpoints) if i not in self.passed_checkpoints]
        if remaining:
            next_cp = self.checkpoints[remaining[0]]
            cp_center = np.array([next_cp.centerx, next_cp.centery])
            to_cp = cp_center - self.car_pos
            forward_vec = np.array([np.cos(self.car_heading), np.sin(self.car_heading)])
            alignment = np.dot(to_cp, forward_vec) / (np.linalg.norm(to_cp) + 1e-6)
            reward += max(0, alignment) * 0.05

        heading_change = abs(self.car_heading - self.prev_heading)
        if heading_change > np.pi / 2:
            reward -= 0.2
        self.prev_heading = self.car_heading

        #if self._on_track(self.car_pos) and throttle > 0:
        #    forward_vec = np.array([np.cos(self.car_heading), np.sin(self.car_heading)])
        #    velocity_vec = self.car_velocity * forward_vec
        #    reward += np.dot(velocity_vec, forward_vec) * 0.01

        if np.linalg.norm(self.car_pos - self.prev_pos) < 1.0:
            reward -= 0.2
        self.prev_pos = self.car_pos.copy()

        self.collided_with_obstacle = False
        for ob in self.obstacles:
            if np.linalg.norm(self.car_pos - ob) < self.obstacle_radius + self.car_radius:
                reward -= 1.0
                terminated = True
                self.collided_with_obstacle = True
                break
        
        if not self._on_track(self.car_pos):
            reward -= 1.0
            terminated = True
        
        for i, cp in enumerate(self.checkpoints):
            if i not in self.passed_checkpoints and cp.collidepoint(*self.car_pos):
                self.passed_checkpoints.add(i)
                reward += 1.0

        if self.finish_line.collidepoint(*self.car_pos) and len(self.passed_checkpoints) == self.num_checkpoints:
            lap_time = time.time() - self.lap_start_time
            self.lap_times.append(lap_time)
            self.lap_start_time = time.time()
            self.passed_checkpoints.clear()
            reward += 5.0

        info.update({
            "current_lap": len(self.lap_times),
            "lap_time": self.lap_times[-1] if self.lap_times else None,
            "checkpoints_passed": len(self.passed_checkpoints),
            "obstacle_collision": self.collided_with_obstacle
        })

        obs = self._get_obs()
        return obs, reward, terminated, False, info

    def _get_obs(self):
        cp_vec = np.zeros(2)
        remaining = [i for i in range(self.num_checkpoints) if i not in self.passed_checkpoints]
        if remaining:
            next_cp = self.checkpoints[remaining[0]]
            cp_center = np.array([next_cp.centerx, next_cp.centery])
            cp_vec = cp_center - self.car_pos

        obs = np.concatenate([
            self.car_pos / np.array([self.screen_width, self.screen_height]),
            [np.cos(self.car_heading), np.sin(self.car_heading)],
            cp_vec / (np.linalg.norm(cp_vec) + 1e-6),
            [self.car_velocity / self.max_velocity]
        ])
        return obs.astype(np.float32)
    
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
        pygame.draw.polygon(self.track_mask, (255, 255, 255), self.track_outline)
        pygame.draw.polygon(self.track_surface, (50, 50, 50), self.track_outline, width=2)

        # Obstacles
        for ob in self.obstacles:
            pygame.draw.circle(screen, (128, 0, 0), ob.astype(int), self.obstacle_radius)

        # Checkpoints
        for i, cp in enumerate(self.checkpoints):
            color = (0, 255, 0) if i in self.passed_checkpoints else(255, 255, 0)
            pygame.draw.rect(screen, color, cp)

        # Finish line
        pygame.draw.rect(screen, (255, 0, 255), self.finish_line)

        if self.debug:
            screen.blit(self.track_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

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