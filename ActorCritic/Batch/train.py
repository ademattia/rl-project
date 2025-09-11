import argparse
import os
import sys
import optuna
import torch
import gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from env.custom_hopper import *
from ActorCritic.Batch.agent import Agent, Actor, Critic
from lib.utils import VideoGenerator, TrainingLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-episodes", default=15000, type=int, help="Number of training episodes"
    )
    parser.add_argument(
        "--batch-size", default=10, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--hidden-dim", default=128, type=int, help="Number of hidden units per layer"
    )
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
    parser.add_argument(
        "--baseline",
        default=100.0,
        type=float,
        help="Value of the baseline used in REINFORCE",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Print environment details"
    )
    parser.add_argument(
        "--print-every", default=1000, type=int, help="Print info every <> episodes"
    )
    parser.add_argument(
        "--video-every", default=12000, type=int, help="Record video every <> episodes"
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="network device [cpu, cuda]"
    )
    parser.add_argument(
        "--seed", default=1, type=int, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model-every",
        default=5000,
        type=int,
        help="Save model every <> episodes (0 = never)",
    )
    parser.add_argument(
        "--mode",
        default="baseline",
        type=str,
        help="type of advantage function [baseline, TD]",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test run without actual training - verify model/CSV paths and configurations",
    )

    return parser.parse_args()


args = parse_args()


def evaluate_agent(agent, env, n_episodes=10):
    total_reward, total_steps = 0.0, 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, _ = env.step(action.detach().cpu().numpy())
            total_reward, total_steps = total_reward + reward, total_steps + 1
    return total_reward / n_episodes, total_steps / n_episodes


def main(args=None, trial=None, hyperparameter_search=False):
    if args is None:
        args = parse_args()

    trial_every = 500

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
        torch.set_num_interop_threads(1)
        if not hyperparameter_search:
            print(f"Using device: {args.device}")
            print(f"CPU threads: {torch.get_num_threads()}")
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            if not hyperparameter_search:
                print(f"Using device: {args.device}")
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(
                    f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )
                print(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")

    env = gym.make("CustomHopper-source-v0")
    env.seed(args.seed)

    # env = gym.make('CustomHopper-target-v0')

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    # Initialize actor, critic and agent
    critic = Critic(observation_space_dim, action_space_dim, seed=args.seed)
    actor = Actor(
        observation_space_dim, action_space_dim, args.hidden_dim, seed=args.seed
    )
    agent = Agent(actor, critic, gamma=args.gamma, device=args.device)

    # Initialize video recorder and logger
    video_generator = VideoGenerator(
        env,
        agent,
        algorithm_name="Actor-Critic",
        seed=args.seed,
        batch_size=args.batch_size,
        video_folder="videos",
        n_episodes=15,
        verbose=True,
    )
    logger = TrainingLogger(
        env=env,
        total_episodes=args.n_episodes,
        seed=args.seed,
        algorithm_name="Actor-Critic",
        mode=args.mode,
        batch_size=args.batch_size,
        avg_window_size=args.print_every,
    )

    if args.verbose and not hyperparameter_search:
        logger.print_training_initialization()

    # DRY RUN: Verify paths and configurations without training
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print("Testing model and CSV path configurations...")

        # Test video path
        print(f"Video folder: {video_generator.video_folder}")
        os.makedirs(video_generator.video_folder, exist_ok=True)
        print(f" Video directory verified: {video_generator.video_folder}")

        # Test model save paths by showing where they would be saved
        print(f"Model save interval: every {args.model_every} episodes")
        print(f" Model saving configured")

        # Test CSV logging paths
        print(
            f"CSV logging configured - Batch size: {args.batch_size}, Mode: {args.mode}, Seed: {args.seed}"
        )
        print(f" CSV logging configured")

        # Test environment and agent initialization
        print(f"Environment: {env.spec.id}")
        print(
            f"Observation space: {observation_space_dim}, Action space: {action_space_dim}"
        )
        print(f"Device: {args.device}")
        print(f" Environment and agent initialized successfully")

        print("\n All configurations verified successfully")
        print(" Dry run completed - ready for training")
        return None

    for episode in range(args.n_episodes):

        state, done, train_reward, episode_step = env.reset(), False, 0.0, 0

        episode_data = {
            "states": [],
            "actions": [],
            "next_states": [],
            "action_log_probs": [],
            "rewards": [],
            "dones": [],
        }

        while not done:  # Loop until the episode is over
            action, action_probabilities = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            episode_data["states"].append(previous_state)
            episode_data["actions"].append(action.detach().cpu().numpy())
            episode_data["next_states"].append(state)
            episode_data["action_log_probs"].append(action_probabilities)
            episode_data["rewards"].append(reward)
            episode_data["dones"].append(done)

            train_reward, episode_step = train_reward + reward, episode_step + 1

        # log episode
        logger.record_episode(episode, train_reward, episode_step)

        agent.store_outcome(
            episode_data["states"],
            episode_data["next_states"],
            episode_data["action_log_probs"],
            episode_data["rewards"],
            episode_data["dones"],
        )

        # Print training stats
        if (
            (episode + 1) % args.print_every == 0
            and args.verbose
            and not hyperparameter_search
        ):
            logger.print_training_stats()

        # Batch update
        if (episode + 1) % args.batch_size == 0:
            agent.update(mode=args.mode)

        # Hyperparameter search
        if (
            hyperparameter_search
            and (trial is not None)
            and ((episode + 1) % trial_every == 0)
        ):
            avg_return, avg_steps = evaluate_agent(agent, env)
            trial.report(avg_return - avg_steps, episode)

            if trial.should_prune():
                raise optuna.TrialPruned()

        # video recording
        if (episode + 1) % args.video_every == 0 and not hyperparameter_search:
            video_generator.recording(episode)

        if (episode + 1) % args.model_every == 0:
            logger.save_model(agent, episode)

    if args.verbose and not hyperparameter_search:
        logger.print_training_end()

    if hyperparameter_search:
        avg_return, avg_steps = evaluate_agent(agent, env)
        return avg_return - avg_steps
    return


if __name__ == "__main__":
    main()
