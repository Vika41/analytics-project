import csv
import gymnasium as gym
import numpy as np
import time

from gymnasium import spaces

class GridBasedEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(GridBasedEnv, self).__init__()
        self.grid_size = (10, 10)
        self.agent_pos = [1, 1]
        self.agent_dir = 0 # 0: up, 1: right, 2: down, 3: left

        self.finish_pos = [1, 1]
        self.prev_pos = self.agent_pos.copy()
        self.num_checkpoints = len(self.checkpoints)

        self.track = np.zeros(self.grid_size, dtype=np.float32)
        self._build_track()
        #self.track[1:-1,1:-1] = 1.0

        obs_dim = 10 * 10 * 2 + self.num_checkpoints + 1
        #self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10, 10, 2), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.max_steps = 200
        self.step_count = 0
        self.lap_start_time = None
        self.lap_times = []
        self.max_laps = 3
        self.current_lap = 0
        self.passed_checkpoints = set()

    def _build_track(self):
        self.track.fill(0.0)
        self.track[1, 1:9] = 1.0
        self.track[1:9, 8] = 1.0
        self.track[8, 2:9] = 1.0
        self.track[2:9, 1] = 1.0

        self.checkpoints = [(1, 4), (4, 8), (8, 6), (6, 1)]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [1, 1]
        self.agent_dir = 1 # Facing right
        self.step_count = 0
        self.lap_start_time = time.time()
        self.lap_times = []
        self.current_lap = 0
        self.passed_checkpoints = set()
        print("[RESET] New episode started")
        obs = self._get_obs()
        info = {"lap_times": self.lap_times, "current_lap": self.current_lap}
        return obs, info
    
    def step(self, action):
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        terminated = False
        lap_time = None

        if action == 0: # Accelerate
            self._move_forward()
        elif action == 1: # Brake
            pass
        elif action == 2: # Turn left
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 3: # Turn right
            self.agent_dir = (self.agent_dir + 1) % 4
        
        #reward = 0.5 if self._on_track() else -1.0
        reward = 0.1
        if self._on_track():
            reward += 1.0
        else:
            reward -= 5.0
            terminated = True
            print(f"[CRASH] Off-track at step {self.step_count}, pos={self.agent_pos}")

        if self.step_count > 1 and self.agent_pos == self.prev_pos:
            reward -= 0.2
        self.prev_pos = self.agent_pos.copy()

        remaining = [i for i in range(self.num_checkpoints) if i not in self.passed_checkpoints]
        if remaining:
            next_cp = self.checkpoints[remaining[0]]
            dist = np.linalg.norm(np.array(next_cp) - np.array(self.agent_pos))
            reward += max(0, 5.0 - dist) * 0.05

        # Checkpoint Logic
        pos_tuple = tuple(self.agent_pos)
        for i, cp in enumerate(self.checkpoints):
            if cp == pos_tuple and i not in self.passed_checkpoints:
                self.passed_checkpoints.add(i)
                reward += 2.0
                print(f"[CHECKPOINT] Passed checkpoint {i}")
        
        # Lap Completion
        if self._lap_completed():
            lap_time = time.time() - self.lap_start_time
            self._log_lap_time(lap_time)
            self.lap_times.append(lap_time)
            self.lap_start_time = time.time()
            self.current_lap += 1
            reward += max(0, 10.0 - lap_time)
            print(f"[LAP] Completed lap {self.current_lap} in {lap_time:.2f}s")
            self.passed_checkpoints.clear()
            if self.current_lap >= self.max_laps:
                terminated = True
                print("[FINISH] Max laps reached")
        
        info = {
            "lap_times": self.lap_times,
            "current_lap": self.current_lap,
            "lap_time": lap_time if self.current_lap > 0 else None,
            "checkpoints_passed": len(self.passed_checkpoints)
        }
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
    
    def _move_forward(self):
        if self.agent_dir == 0: # Up
            self.agent_pos[0] -= 1
        elif self.agent_dir == 1: # Right
            self.agent_pos[1] += 1
        elif self.agent_dir == 2: # Down
            self.agent_pos[0] += 1
        elif self.agent_dir == 3: # Left
            self.agent_pos[1] -= 1

    def _on_track(self):
        x, y = self.agent_pos
        return 0 <= x < 10 and 0 <= y < 10 and self.track[x, y] == 1.0
    
    def _lap_completed(self):
        #return self.agent_pos == self.finish_pos
        return tuple(self.agent_pos) == tuple(self.finish_pos) and len(self.passed_checkpoints) == len(self.checkpoints)
    
    def _log_lap_time(self, lap_time):
        with open("grid_lap_times.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.current_lap, lap_time])
    
    def _get_obs(self):
        layout = np.copy(self.track)
        agent_mask = np.zeros_like(layout)
        x, y = self.agent_pos
        if 0 <= x < 10 and 0 <= y < 10:
            agent_mask[x, y] = 1.0
        checkpoint_obs = np.zeros(self.num_checkpoints, dtype=np.float32)
        for i in self.passed_checkpoints:
            checkpoint_obs[i] = 1.0
        lap_ratio = np.array([self.current_lap / self.max_laps], dtype=np.float32)
        #obs = np.stack([layout, agent_mask], axis=-1)
        obs = np.concatenate([
            layout.flatten(),
            agent_mask.flatten(),
            checkpoint_obs,
            lap_ratio
        ]).astype(np.float32)
        return obs
    
    def render(self, mode='human'):
        print(f"Position: {self.agent_pos}, Direction: {self.agent_dir}, Lap: {self.current_lap}, Checkpoints: {sorted(self.passed_checkpoints)}")

    def close(self):
        #return super().close()
        pass
    
if __name__ == "__main__":
    env = GridBasedEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    env.close()