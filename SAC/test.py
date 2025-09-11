import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch.nn as nn
import gym
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import random
import torch

# Global seed setting function
def set_seed(seed: int, env: gym.Env = None):
    # Seed Python
    random.seed(seed)
    
    # Seed NumPy
    np.random.seed(seed)
    
    # Seed PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Seed Gym environment
    env.seed(seed)


def evaluate_model(model, venv, n_episodes: int = 20, eval_env=None, deterministic: bool = True):
    print("Dynamics parameters:", eval_env.get_parameters())

    episode_rewards, _ = evaluate_policy(
        model,
        venv,
        n_eval_episodes=n_episodes,
        deterministic=deterministic,
        return_episode_rewards=True
    )

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"Mean episode return: {mean_reward:.2f} ± {std_reward:.2f}")

    return mean_reward, std_reward


def main():
    
    # Create target environment
    env_name = "CustomHopper-source-v0"
    eval_env = gym.make(env_name)
    # eval_env.set_scale(np.array([1.11442929, 1.08946531, 0.98016055]))
    # eval_env.set_masses()
    eval_base = Monitor(eval_env)
    eval_venv = DummyVecEnv([lambda: eval_base])

    # Set seeds
    seed = 15
    set_seed(seed, eval_env)

    # Load VecNormalize 
    
    vecnorm_filename = "SAC_CustomHopper-source-v0_vecnorm_seed15_NoUDR_step200000.pkl"
    vecnorm_path = os.path.join("..", "SAC", "models", vecnorm_filename)
    vecnorm = VecNormalize.load(vecnorm_path, eval_venv)
    vecnorm.training = False
    vecnorm.norm_reward = False
    
    
    # Load model 
    model_path = os.path.join("..", "SAC", "models", "SAC_CustomHopper-source-v0_seed15_NoUDR_step200000.zip")
    model = SAC.load(model_path, env=vecnorm, device="cpu")

    # Evaluate the policy
    print("Evaluating on environment:", env_name)
    evaluate_model(model, vecnorm, n_episodes=50, eval_env=eval_env, deterministic=True)

    vecnorm.close()
    
    
    
if __name__ == '__main__':
	main()
