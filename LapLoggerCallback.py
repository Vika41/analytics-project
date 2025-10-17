import csv
import os

from stable_baselines3.common.callbacks import BaseCallback

class LapLoggerCallback(BaseCallback):
    def __init__(self, log_path="lap_stats.csv", verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.lap_data = []

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[-1]

        if "checkpoints_passed" in info:
            self.logger.record("/checkpoints_passed", info["checkpoints_passed"])
        if "current_lap" in info:
            self.logger.record("/lap", info["current_lap"])
        if "lap_time" in info and info["lap_time"] is not None:
            self.logger.record("/lap_time", info["lap_time"])
            self.lap_data.append((self.num_timesteps, info["current_lap"], info["lap_time"]))
            #lap_time = info["lap_time"]
            #lap_num = info.get("current_lap", -1)
            #self.lap_data.append((self.num_timesteps, lap_num, lap_time))
            if self.verbose:
                #print(f"[CALLBACK] Lap {lap_num} at step {self.num_timesteps}: {lap_time:.2f}s")
                print(f"[CALLBACK] Lap {info['current_lap']} at step {self.num_timesteps}: {info['lap_time']:.2f}s")
        return True

    def _on_training_end(self) -> None:
        dir_path = os.path.dirname(self.log_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestep", "Lap", "LapTime"])
            writer.writerows(self.lap_data)
        if self.verbose:
            print(f"[CALLBACK] Saved lap stats to {self.log_path}")