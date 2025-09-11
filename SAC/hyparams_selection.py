import os
import gym
import optuna
import torch.nn as nn
import numpy as np
import os
import sys
import xml.etree.ElementTree as ET
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env.custom_hopper import *
from SAC.callbacks import Logger

# Objective function for Optuna
def objective(trial):

    seed = 10
    env_name = "CustomHopper-source-v0"

    # Crea ambiente
    base = gym.make(env_name, udr=False)
    base.seed(seed)
    base = Monitor(base)
    train_venv = DummyVecEnv([lambda: base])
    vecnorm = VecNormalize(train_venv, norm_obs=True, norm_reward=True)

    # Hyperparameter space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    tau = trial.suggest_float("tau", 0.005, 0.02)
    gamma = trial.suggest_float("gamma", 0.95, 0.995)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    net_arch = trial.suggest_categorical("net_arch", [(64, 64), (128, 128), (256, 256)])
    
    policy_kwargs = dict(
    net_arch=list(net_arch),
    activation_fn=nn.ReLU
    )

    model = SAC(
        policy="MlpPolicy",
        env=vecnorm,
        learning_rate=learning_rate,
        buffer_size=500_000,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=1,
        gradient_steps=1,
        learning_starts=5_000,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cpu"
    )

    # Train for a short period to evaluate
    model.learn(total_timesteps=50_000)

    # Evaluate the mean reward over 5 episodes- mean is the metric to maximize
    mean_reward, _ = evaluate_policy(model, vecnorm, n_eval_episodes=5, deterministic=True)

    vecnorm.close()
    return mean_reward

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # number of trials

    print("Best hyperparameters found:")
    print(study.best_params)
    print(f"Best mean reward: {study.best_value}")

if __name__ == "__main__":
    main()