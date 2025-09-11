import os
import time 
import csv
import sys
import xml.etree.ElementTree as ET

import numpy as np
import gym
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env.custom_hopper import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from distributions import BaseMassDistribution, UniformMassDistribution, NormalMassDistribution, DegenerateMassDistribution

class Logger(BaseCallback):
    def __init__(self, env_name: str, seed: int, udr: bool, log_freq: int = 10_000, video_freq: int = 50_000, model_freq: int = 50_000, distribution: BaseMassDistribution = None, save_model: bool = True, log_dir: str = "results", 
                  model_dir: str = "models", video_dir: str = "videos", n_eval_episodes: int = 10, total_timesteps=None, verbose=0):
        super().__init__()
        self.env_name, self.seed = env_name, seed
        self.udr = udr
        self.save_model = save_model
        self.model_freq = model_freq
        self.udr_str = "UDR" if self.udr else "NoUDR"
        self.log_freq, self.video_freq = log_freq, video_freq
        self.log_path, self.log_dir = None, log_dir
        self.model_dir = model_dir
        self.video_dir = video_dir
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose
        self.total_timesteps = total_timesteps
        self.best_mean_reward = -float("inf")
        self.start_time = None
        self.inner_env = gym.make(env_name, udr=udr)
        
        if distribution is None:
            self.dist = DegenerateMassDistribution()
        else:
            self.dist = distribution
            
        if self.udr: 
            self.inner_env.set_distribution(self.dist)
            
        self.stats_base = Monitor(self.inner_env)
        self.stats_env = DummyVecEnv([lambda: self.stats_base])
        
        self._path_init_() 
        

    def _path_init_(self) -> None:

        self.log_path = os.path.join("results", f"SAC_{self.env_name}_seed{self.seed}_{self.udr_str}_results.csv")
        os.makedirs("results", exist_ok=True)

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "mean_reward", "std_reward", "mean_steps"])
            
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


    def _on_training_start(self) -> None:
        self.start_time = time.time()
        if self.verbose:
            udr_str = "UDR" if self.udr else "NoUDR"
            print("\n" + "═"*80)
            print(f" Evaluation Initialization - SAC {self.env_name} seed {self.seed} {udr_str}")
            print("═"*80)
            print(f" Episodes per eval : {self.n_eval_episodes}")
            print(f" Eval frequency    : {self.log_freq} steps")
            print(f" Log file          : {self.log_path}")
            print("═"*80 + "\n")

    def model_save_(self, mean_reward) -> None:
        model_path = os.path.join(self.model_dir,
            f"SAC_{self.env_name}_seed{self.seed}_{self.udr_str}_step{self.num_timesteps}.zip")

        vecnorm_path = os.path.join(self.model_dir,
            f"SAC_{self.env_name}_vecnorm_seed{self.seed}_{self.udr_str}_step{self.num_timesteps}.pkl")
        self.model.get_env().save(vecnorm_path)
        self.model.save(model_path)

        is_best = mean_reward > self.best_mean_reward
        if is_best:
            self.best_mean_reward = mean_reward
            best_model_path = os.path.join(
                self.model_dir,
                f"SAC_{self.env_name}_seed{self.seed}_{self.udr_str}_best_model.zip"
            )
            self.model.save(best_model_path)
            best_vecnorm_path = os.path.join(
                self.model_dir,
                f"SAC_{self.env_name}_seed{self.seed}_{self.udr_str}_best_vecnorm.pkl"
            )
            self.model.save(best_vecnorm_path)

        if self.verbose:
            bar_len = 30
            progress = (self.num_timesteps / getattr(self.model, "_total_timesteps", 1))
            filled = int(bar_len * progress)
            bar = "#" * filled + "─" * (bar_len - filled)

            print("\n" + "═"*80)
            print(f" Checkpoint at step {self.num_timesteps:,} "
                    f"[{bar}] {progress*100:5.1f}%")
            print("─"*80)
            print(f" Saved Model   : {model_path}")
            print(f" Saved VecNorm  : {vecnorm_path}")
            if is_best:
                print(f" New Best Model saved : {best_model_path}")
                print(f" New Best VecNorm saved : {best_vecnorm_path}")
            print("═"*80 + "\n")
        
    def _on_step(self) -> bool: 
        log = (self.num_timesteps % self.log_freq == 0)
        video = (self.num_timesteps % self.video_freq == 0)
        model = (self.num_timesteps % self.model_freq == 0) and self.save_model
        if log:

            if video: 
                video_path = os.path.join(
                    self.video_dir,f"SAC_{self.env_name}_seed{self.seed}_{self.udr_str}_step{self.num_timesteps:09d}.mp4") 
                video_recorder = VideoRecorder(self.inner_env, video_path, enabled=True)
                
                if self.verbose: 
                    print("\n" + "═"*80)
                    print(" Video Callback Recording Started")
                    print("═"*80)
                    print(f" Step         : {self.num_timesteps:,}")
                    print(f" Save Path    : {video_path}")
                    print(f" Eval Episodes: {self.n_eval_episodes}")
                    print("═"*80)
                
                
            self.model.get_env().save("models/vecnorm.pkl")
            self.stats_vec = VecNormalize.load("models/vecnorm.pkl", self.stats_env)
            self.stats_vec.training, self.stats_vec.norm_reward = False, False


            rewards, lengths = [], []
            for _ in range(self.n_eval_episodes):
                state = self.stats_vec.reset()
                done = False
                ep_reward, ep_steps = 0.0, 0
                
                while not done:
                    action, _ = self.model.predict(state, deterministic=True)
                    state, reward, done, _ = self.stats_vec.step(action)
                    ep_reward, ep_steps = ep_reward + reward, ep_steps + 1
                    if video: 
                        self.inner_env.render()
                        video_recorder.capture_frame()
                rewards.append(ep_reward)
                lengths.append(ep_steps)

            mean_reward, std_reward = np.mean(rewards), np.std(rewards)
            avg_steps = np.mean(lengths)
            avg_r_per_step = mean_reward / avg_steps if avg_steps > 0 else 0.0
            
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.num_timesteps, mean_reward, std_reward, avg_steps])

            if video: 
                video_recorder.close()
                video_recorder.enabled = False  
                if self.verbose:
                    print("\n" + "═"*80)
                    print(" Video Callback Recording Finished")
                    print("═"*80)
                    print(f" Saved File   : {video_path}")
                    print("═"*80 + "\n") 

            if model: 
                self.model_save_(mean_reward)    
            
            
            if self.verbose: 
                elapsed = time.time() - self.start_time
                progress = (self.num_timesteps / self.total_timesteps) if self.total_timesteps else 0
                bar_len = 30
                filled = int(bar_len * progress)
                bar = "#" * filled + "─" * (bar_len - filled)
                udr_str = "UDR" if self.udr else "NoUDR"

                print("\n" + "═"*80)
                print(f" Evaluation "
                    f"at {self.num_timesteps:,}/{self.total_timesteps:,} steps "
                    f"[{bar}] {progress*100:5.1f}%")
                print("─"*80)
                print(f" Mean Return    : {mean_reward:8.2f} ± {std_reward:.2f}")
                print(f" Avg Steps      : {avg_steps:8.1f}")
                print(f" Avg Reward/Step: {avg_r_per_step:8.3f}")
                print("─"*80)
                print(f" Elapsed Time   : {elapsed/60:6.1f} min")
                print("═"*80 + "\n")
        
        return True 
    
    
    def _on_training_end(self) -> None:
        if self.verbose:
            total_time = time.time() - self.start_time if self.start_time else 0
            udr_str = "UDR" if self.udr else "NoUDR"
            print("\n" + "═"*80)
            print(f" Evaluation Completed - SAC {self.env_name} seed {self.seed} {udr_str}")
            print("═"*80)
            print(f" Total Timesteps : {self.num_timesteps:,}")
            print(f" Total Time      : {total_time/60:.1f} minutes")
            print(f" Log File Saved  : {self.log_path}")
            print(f" Results Folder  : {os.path.dirname(self.log_path)}")
            print("═"*80 + "\n")



    