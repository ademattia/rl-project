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
from callbacks import Logger



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

def main():
    seed = 49
    set_seed(seed)
    
    # Training parameters
    total_timesteps = 200_000
    env_name = "CustomHopper-source-v0"
    udr = False
    udr_str = "UDR" if udr else "NoUDR"

    # create train environment
    base = gym.make(env_name, udr=udr)
    base.seed(seed)
    base = Monitor(base)
    train_venv = DummyVecEnv([lambda: base])
    vecnorm = VecNormalize(train_venv, norm_obs=True, norm_reward=True)
    

    # network architecture
    policy_kwargs = dict(
        net_arch=[128, 128])

    # create the model
    model = SAC(
        policy="MlpPolicy",
        env=vecnorm,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=64,
        tau=0.01,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=5_000,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        verbose=0,
        device="cpu"
    )
    
    

    # callback for logging
    logger = Logger(env_name="CustomHopper-target-v0" , seed=seed, udr=udr, log_freq = 10_000, video_freq = 500_000, model_freq=20_000, total_timesteps=200_000, verbose=1)

    # model training
    model.learn(total_timesteps=total_timesteps, callback=logger)
  
if __name__ == '__main__':
	main()
