import os
import time
import csv
import numpy as np

import torch
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

DEFAULT_VIDEO_PATH = "videos"
DEFAULT_RESULTS_PATH = "results"
DEFAULT_MODELS_PATH = "models"


class VideoGenerator:
    def __init__(
        self,
        env,
        agent,
        n_episodes: int,
        seed: str,
        algorithm_name: str,
        batch_size: str,
        video_folder=DEFAULT_VIDEO_PATH,
        verbose=True,
    ):
        self.env = env
        self.agent = agent
        self.video_folder = video_folder
        self.n_episodes = n_episodes
        self.verbose = verbose

        self.seed = seed
        self.batch_size = batch_size
        self.algorithm_name = algorithm_name

        if not os.path.exists(self.video_folder):
            os.makedirs(self.video_folder)

    def recording(self, episode_id):
        safe_alg = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in str(self.algorithm_name)
        )
        video_path = os.path.join(
            self.video_folder,
            f"{safe_alg}_batch{self.batch_size}_seed{self.seed}_episode{(episode_id+1):06d}.mp4",
        )

        video_recorder = VideoRecorder(self.env, video_path, enabled=True)

        if self.verbose:
            print("\n" + "═" * 80)
            print(" Video Recording Started")
            print("═" * 80)
            print(f" Episode ID   : {episode_id}")
            print(f" Save Path    : {video_path}")
            print("═" * 80 + "\n")

        for _ in range(self.n_episodes):
            done = False
            state = self.env.reset()

            while not done:
                with torch.no_grad():
                    action, _ = self.agent.get_action(state, evaluation=True)
                state, _, done, _ = self.env.step(action.detach().cpu().numpy())
                self.env.render()
                video_recorder.capture_frame()

        video_recorder.close()
        video_recorder.enabled = False

        if self.verbose:
            print("\n" + "═" * 80)
            print(" Video Recording Finished")
            print("═" * 80)
            print(f" Saved File   : {video_path}")
            print("═" * 80 + "\n")


class TrainingLogger:
    def __init__(
        self,
        env,
        total_episodes: str,
        seed: str,
        algorithm_name: str,
        batch_size: str,
        baseline=None,
        mode=None,
        results_dir=DEFAULT_RESULTS_PATH,
        models_dir=DEFAULT_MODELS_PATH,
        avg_window_size=50,
    ):
        self.env = env
        self.episode_returns = []
        self.episode_lengths = []
        self.start_time = time.time()

        self.results_dir = results_dir
        self.models_dir = models_dir
        self.batch_size = batch_size
        self.baseline = baseline
        self.mode = mode
        self.total_steps = 0
        self.episode = 0
        self.total_episodes = total_episodes
        self.seed = seed
        self.avg_window_size = avg_window_size
        self.best_reward = -np.inf
        self.safe_alg = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in str(algorithm_name)
        )

        if self.safe_alg != "REINFORCE" and self.safe_alg != "Actor-Critic":
            raise ValueError(f"Algorithm {self.safe_alg} NOT supported.")

        if self.safe_alg == "Actor-Critic":
            self.safe_mode = "".join(
                c if c.isalnum() or c in "-_." else "_" for c in str(mode)
            )
            self.mode = self.safe_mode

        # check if directories exist, if not create them
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)

        # create results file path
        tmp = f"{self.safe_alg}_batch{self.batch_size}_seed{self.seed}"
        if self.safe_alg == "REINFORCE":
            tmp += f"_baseline_{self.baseline}_stats.csv"
        elif self.safe_alg == "Actor-Critic":
            tmp += f"_mode_{self.safe_mode}_stats.csv"

        self.filepath = os.path.join(
            self.results_dir,
            tmp,
        )

        # Initialize CSV
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward", "Episode Steps"])

    # ----------------------------
    # Logging
    # ----------------------------
    def record_episode(self, episode, train_return, episode_steps):
        self.episode_returns.append(train_return)
        self.episode_lengths.append(episode_steps)
        self.total_steps += episode_steps
        self.episode += 1

        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([(episode + 1), train_return, episode_steps])

    # ----------------------------
    # Printing
    # ----------------------------
    def print_training_initialization(self):
        print("\n" + "═" * 80)
        print(" Training Initialization")
        print("═" * 80)
        print(f" Seed              : {self.seed}")
        print(f" Action space      : {self.env.action_space}")
        print(f" State space       : {self.env.observation_space}")
        if hasattr(self.env, "get_parameters"):
            print(f" Dynamics params   : {self.env.get_parameters()}")
        print("═" * 80 + "\n")

    def print_training_stats(self):
        elapsed = time.time() - self.start_time
        episode_returns = self.episode_returns[-self.avg_window_size :]
        episode_steps = self.episode_lengths[-self.avg_window_size :]
        episode = self.episode

        train_reward = self.episode_returns[-1]
        steps = self.episode_lengths[-1]
        avg_reward = np.mean(episode_returns)
        std_reward = np.std(episode_returns)
        avg_steps = np.mean(episode_steps)

        progress = (episode + 1) / self.total_episodes
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "#" * filled + "─" * (bar_len - filled)

        print("\n" + "═" * 80)
        print(
            f" Episode {episode+1:,}/{self.total_episodes} "
            f"[{bar}] {progress*100:5.1f}%"
        )
        print("─" * 80)
        print(f" Last Return    : {train_reward:8.2f}")
        print(f" Last Steps     : {steps:8d}")
        print(f" Reward/Step    : {train_reward/steps:8.3f}")
        print("─" * 80)
        print(f" Avg Return     : {avg_reward:8.2f} ± {std_reward:.2f}")
        print(f" Avg Steps      : {avg_steps:8.1f}")
        print(f" Avg Reward/Step: {avg_reward/avg_steps:8.3f}")
        print("─" * 80)
        print(f" Elapsed Time   : {elapsed/60:6.1f} min")
        print("═" * 80 + "\n")

    def print_training_end(self):
        total_time = time.time() - self.start_time
        print("\n" + "═" * 80)
        print(" Training Completed")
        print("═" * 80)
        print(f" Total Episodes : {self.total_episodes}")
        print(f" Total Time     : {total_time/60:.1f} minutes")
        print("═" * 80 + "\n")

    # ----------------------------
    # Model saving
    # ----------------------------
    def save_model(self, agent, episode=None):

        tmp = f"{self.safe_alg}_batch{self.batch_size}"
        if self.safe_alg == "REINFORCE":
            tmp += f"_baseline_{self.baseline}"
        elif self.safe_alg == "Actor-Critic":
            tmp += f"_mode_{self.safe_mode}"
        tmp += f"_seed{self.seed}"

        safe_episode = (
            f"{tmp}_episode{(episode+1):06d}.mdl" if episode is not None else ""
        )
        save_path = os.path.join(self.models_dir, safe_episode)

        if self.safe_alg == "REINFORCE":
            torch.save(agent.policy.state_dict(), save_path)
        elif self.safe_alg == "Actor-Critic":
            torch.save(agent.actor.state_dict(), save_path)

        print("\n" + "═" * 80)
        print(" Model Saved")
        print("═" * 80)
        print(f" Episode       : {episode}")
        print(f" Save Path     : {save_path}")
        print("═" * 80 + "\n")

        # save best model
        episode_returns = self.episode_returns[-self.avg_window_size :]
        avg_reward = np.mean(episode_returns)

        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            best_path = os.path.join(self.models_dir, f"{tmp}_best.mdl")

            if self.safe_alg == "REINFORCE":
                torch.save(agent.policy.state_dict(), best_path)
            elif self.safe_alg == "Actor-Critic":
                torch.save(agent.actor.state_dict(), best_path)

            print("\n" + "═" * 80)
            print(" New Best Model Saved")
            print("═" * 80)
            print(f" Episode       : {episode}")
            print(f" Avg Reward    : {avg_reward:.2f}")
            print(f" Save Path     : {best_path}")
            print("═" * 80 + "\n")
