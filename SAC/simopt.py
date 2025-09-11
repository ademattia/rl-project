import os
import sys
import argparse
import numpy as np
import random
import torch
from pathlib import Path

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env.custom_hopper import *
from callbacks import Logger


# Global seed setting function
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_action_sac(model, obs, vecnorm):
    obs_norm = vecnorm.normalize_obs(obs[None, :])
    action, _ = model.predict(obs_norm, deterministic=True)
    return action

def rollouts(env, model, vecnorm, episodes=1, max_steps=500):
    state_shape = env.observation_space.shape[0]
    trajectory = np.empty((max_steps, state_shape), dtype=np.float32)
    ptr = 0
    done, state = False, env.reset()
    trajectory[ptr] = state
    steps = 0
    obs_buffer = np.empty(state_shape - 1, dtype=np.float32)
    
    while not done and steps < max_steps-1:
        obs_buffer[:] = state[1:]
        action = get_action_sac(model, obs_buffer, vecnorm)
        state, _, done, _ = env.step(action)
        ptr += 1
        if not done:
            trajectory[ptr] = state
        steps += 1
    
    return trajectory[:ptr]

def step_discrepancy(trajectory, env, model, vecnorm, scale):
    env.set_scale(scale)
    env.set_masses()
    traj_len = len(trajectory) - 1
    if traj_len <= 0:
        return 0.0

    discrepancies = np.empty(traj_len, dtype=np.float32)
    obs_buffer = np.empty(trajectory.shape[1] - 1, dtype=np.float32)
    
    for i in range(traj_len):
        state = trajectory[i]
        next_state = trajectory[i+1]
        qpos, qvel = state[:env.model.nq], state[env.model.nq:]
        obs_buffer[:] = state[1:]
        env.set_state(qpos, qvel)
        action = get_action_sac(model, obs_buffer, vecnorm)
        env.do_simulation(action, env.frame_skip)
        state_pred = np.concatenate([env.sim.data.qpos.flat, env.sim.data.qvel.flat])
        diff = state_pred - next_state
        discrepancies[i] = np.dot(diff, diff)
    
    return np.mean(discrepancies)


def sample_candidates(mu, cov, n):
    return np.random.multivariate_normal(mu, cov, size=n)

def compute_weights_from_rewards(rewards, eta):
    maxR = np.max(rewards)
    ex = np.exp((rewards - maxR) / (eta + 1e-12))
    w = ex / (np.sum(ex) + 1e-12)
    return w

def weighted_gaussian_update(thetas, weights, sigma_floor):

    mu = (weights[:, None] * thetas).sum(axis=0)
    diff = thetas - mu[None, :]
    cov = (weights[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0)

    cov = 0.5 * (cov + cov.T)

    d = cov.shape[0]
    for i in range(d):
        if cov[i, i] < sigma_floor[i]**2:
            cov[i, i] = sigma_floor[i]**2
    return mu, cov


def simopt_reps_then_train(
    seed=0,
    reps_iter=5,
    n_points=200,
    n_trajs=8,
    eta_scale=0.5,
    sigma_init=0.15,
    sigma_floor_frac=1e-3,
    final_timesteps=200_000,
    save_dir="outputs"
):
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    env_source_id = "CustomHopper-source-v0"
    env_target_id = "CustomHopper-target-v0"

    env_target = gym.make(env_target_id, full_obs=True)
    env_target.seed(seed)
    
    # Initialize policy and agent
    eval_env = gym.make(env_target_id)
    eval_base = Monitor(eval_env)
    eval_venv = DummyVecEnv([lambda: eval_base])
 
    # Load VecNormalize and model
    vecnorm_filename = os.path.join("..", "SAC", "models", "SAC_CustomHopper-source-v0_vecnorm_seed10_NoUDR_step160000.pkl")
    vecnorm = VecNormalize.load(vecnorm_filename, eval_venv)
    vecnorm.training = False
    vecnorm.norm_reward = False
    model_path = os.path.join("..", "SAC", "models", "SAC_CustomHopper-source-v0_seed10_NoUDR_step160000.zip")
    model = SAC.load(model_path, env=vecnorm, device="cpu")
   
    # Collect target trajectories
    target_trajs = [rollouts(env_target, model, vecnorm, episodes=1) for _ in range(n_trajs)]

    d = 3
    mu = np.ones(d)            
    cov = np.eye(d) * (sigma_init**2)
    sigma_floor = np.ones(d) * (sigma_floor_frac)  

    param_range = 1.0  
    sigma_floor = np.maximum(sigma_floor, 1e-3 * param_range) 

    # SimOpt REPS loop
    for it in range(reps_iter):
        print(f"\n=== REPS iter {it+1}/{reps_iter} ===")
        # sample candidates
        candidates = sample_candidates(mu, cov, n_points)   
        # clip to plausible range
        candidates = np.clip(candidates, 0.3, 1.8)

        discrepancies = np.zeros(n_points, dtype=np.float64)

        # evaluate each candidate: use step_discrepancy averaged over target_trajs
        for i in range(n_points):
            theta = candidates[i]
            # evaluate discrepancy on each stored target trajectory
            
            dvals = []
            env_sim = gym.make(env_source_id, full_obs=True)
            env_sim.seed(seed)
            dvals = []
            for traj in target_trajs:   # target_trajs è una lista di traiettorie
                dval = step_discrepancy(traj, env_sim, model, vecnorm, scale=theta)
                dvals.append(dval)
            discrepancies[i] = np.mean(dvals)
            env_sim.close()
            if (i+1) % 50 == 0:
                print(f" evaluated {i+1}/{n_points} candidates")

        # REWARD is negative discrepancy
        rewards = -discrepancies

        # choose eta adaptively. 
        rng = np.max(rewards) - np.min(rewards)
        eta = eta_scale * (rng + 1e-8)
        if eta <= 0:
            eta = 1e-3

        weights = compute_weights_from_rewards(rewards, eta)

        # update gaussian via weighted MLE
        mu_new, cov_new = weighted_gaussian_update(candidates, weights, sigma_floor)

        # regularize update: do not reduce covariance too fast
        shrink = 0.7
        cov = shrink * cov + (1 - shrink) * cov_new
        mu = mu_new

        print(f"iter {it+1} | eta={eta:.4e} | mu = {mu} | tr mean discrepancy = {np.mean(discrepancies):.6f}")

    final_theta = mu.copy()
    print("\nFinal estimated theta (scale factors):", final_theta)

    print("Starting final SAC training with estimated simulator parameters...")

    # Build train environment with final theta applied
    env_train = gym.make(env_source_id)
    env_train.seed(seed)
    env_train.set_scale(final_theta)
    env_train.set_masses()


    env_train = Monitor(env_train)
    train_venv = DummyVecEnv([lambda: env_train])
    vecnorm = VecNormalize(train_venv, norm_obs=True, norm_reward=True)
    # policy kwargs
    policy_kwargs = dict(net_arch=[128, 128])
    model_final = SAC(policy="MlpPolicy",
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
        device="cpu")

    logger = Logger(env_name="CustomHopper-target-v0" , seed=300, udr=False, log_freq = 10_000, video_freq = 500_000, model_freq=50_000, total_timesteps=200_000, verbose=1)

    model_final.learn(total_timesteps=final_timesteps, callback=logger)

    eval_env = gym.make(env_target_id)
    eval_env.seed(seed)

    
    mean_ret, std_ret = evaluate_policy(model_final, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Evaluation on target env -> mean_return: {mean_ret:.2f} std: {std_ret:.2f}")

    # save model
    out_model_path = os.path.join(save_dir, f"sac_final_reps_seed{seed}.zip")
    model_final.save(out_model_path)
    print("Saved final model to", out_model_path)

    # close envs
    env_train.close()
    env_target.close()
    eval_env.close()

    return final_theta, model_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reps", type=int, default=5, help="REPS iterations")
    parser.add_argument("--n_points", type=int, default=200, help="samples per iteration")
    parser.add_argument("--n_trajs", type=int, default=8, help="target trajectories to use for discrepancy")
    parser.add_argument("--final_timesteps", type=int, default=200000)
    args = parser.parse_args()

    simopt_reps_then_train(
        seed=args.seed,
        reps_iter=args.reps,
        n_points=args.n_points,
        n_trajs=args.n_trajs,
        final_timesteps=args.final_timesteps,
        save_dir="simopt_outputs"
    )
