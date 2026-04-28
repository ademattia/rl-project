# Reinforcement Learning in Robotics and Sim-to-Real Transfer

A reinforcement learning study on the MuJoCo Hopper, a one-legged robot that must learn to walk forward by controlling three rotational actuators. Three algorithms of increasing complexity are implemented and compared — REINFORCE, batch Actor-Critic (with both Monte Carlo baseline and one-step TD advantages), and Soft Actor-Critic (SAC) — to highlight how policy gradient methods scale from textbook formulations to state-of-the-art entropy-regularized off-policy approaches.

The project then addresses the *sim-to-real gap*: the difficulty of transferring policies learned in simulation to a system whose dynamics differ from the simulator. A *source* environment, in which the torso mass is shifted by −30%, stands in for the simulator; a *target* environment with nominal masses stands in for the real system. Two strategies are evaluated: Uniform Domain Randomization (UDR), which trains on a fixed distribution of perturbed dynamics, and a SimOpt-style adaptive scheme based on Relative Entropy Policy Search (REPS), which iteratively refines that distribution by minimizing a transition-based discrepancy between simulated and real rollouts collected with the current policy.

The codebase is organized so that each algorithm lives in its own self-contained module sharing a common environment and a small library of training utilities. A parallel orchestrator coordinates multi-seed, multi-configuration experimental sweeps, and Optuna is integrated for hyperparameter selection. A separate report (PDF) accompanies the repository and contains the full methodology, experimental setup, and discussion of results.

Developed as a project for the **Machine Learning and Deep Learning** course (Prof. Barbara Caputo, 2025) at Politecnico di Torino.

## Demo

Trained SAC policy walking on the source environment (200k training steps, seed 10):

![SAC Hopper walking](assets/sac_hopper_walking.gif)

Full-length sample videos for each algorithm and configuration are available under [`videos sample/`](videos%20sample/).

## Repository structure

```
RL-Project/
├── env/                      # Custom Hopper environment
│   ├── custom_hopper.py      # Source/target variants, UDR, mass scaling
│   ├── mujoco_env.py         # MuJoCo → Gym wrapper
│   └── assets/hopper.xml     # Robot model
├── REINFORCE/                # Vanilla policy gradient
│   ├── agent.py              # Policy network + Agent (with constant baseline)
│   ├── train.py              # Single training run
│   ├── main.py               # Sequential sweep over seeds/baselines
│   ├── run_REINFORCE.py      # Parallel orchestrator (multi-process)
│   ├── hparams_selection.py  # Optuna hyperparameter search
│   └── test.py               # Load and evaluate a saved model
├── ActorCritic/Batch/        # Batch Actor-Critic
│   ├── agent.py              # Actor + Critic, "baseline" and "TD" advantages
│   ├── train.py / main.py / run_ActorCritic.py   # As above
├── SAC/                      # Soft Actor-Critic (Stable-Baselines3)
│   ├── train.py              # SAC training with VecNormalize
│   ├── test.py               # Evaluation on source/target
│   ├── callbacks.py          # Custom logger: eval, video, checkpoints
│   ├── distributions.py      # Mass distributions for UDR
│   ├── hyparams_selection.py # Optuna search
│   └── simopt.py             # SimOpt + REPS pipeline
├── lib/                      # Shared utilities
│   ├── utils.py              # VideoGenerator, TrainingLogger
│   └── plot_graph.py         # Plotting (mean ± min/max bands)
├── outcmaes/                 # CMA-ES output (alternative tried for SimOpt)
├── videos sample/            # Example training/evaluation videos
├── test_random_policy.py     # Sanity-check script (random policy)
└── requirements.txt
```

The three algorithm folders are intentionally self-contained, so each one can be inspected and run independently. Common code lives in `lib/` and `env/`.

## Installation

The project depends on `mujoco-py`, which requires MuJoCo 2.1 installed on the system. Tested with Python 3.8.

```bash
git clone https://github.com/ADemattia/RL-Project.git
cd RL-Project
pip install -r requirements.txt
```

If `mujoco-py` fails to build, refer to the [official installation guide](https://github.com/openai/mujoco-py). On Linux, `patchelf` (already in the requirements) is needed.

To verify the environment works:

```bash
python test_random_policy.py
```

This should open a viewer with the Hopper falling repeatedly under a random policy.

## Reproducing the experiments

### REINFORCE

Single run with custom hyperparameters:

```bash
cd REINFORCE
python train.py --n-episodes 15000 --batch-size 10 --baseline 20.0 --seed 15 --verbose
```

Full sweep used in the report (3 seeds × {batch 1, 10, 50} × {baseline 0, 20}):

```bash
python run_REINFORCE.py --seeds 8 15 17 --batch-sizes 1 10 50 --baselines 0.0 20.0 --max-procs 4
```

Hyperparameter search (Optuna):

```bash
python hparams_selection.py
```

Best configuration found: `gamma=0.99`, `baseline=20.0`, `batch_size=10`.

### Actor-Critic

Single run, choosing the advantage variant via `--mode`:

```bash
cd ActorCritic/Batch
python train.py --n-episodes 15000 --batch-size 10 --mode TD --seed 15 --verbose
# or --mode baseline for V(s)-based advantages
```

Full sweep:

```bash
python run_ActorCritic.py --seeds 8 15 17 --batch-sizes 1 10 50 --modes baseline TD
```

### SAC

Standard training on the source environment (200,000 steps):

```bash
cd SAC
python train.py
```

Hyperparameters and the seed are set inside `train.py` (`seed=49`, `udr=False` by default). Edit the file or wrap with a small driver to sweep configurations. Best hyperparameters found via Optuna: `lr=3e-4`, `tau=0.01`, `gamma=0.995`, `batch_size=64`, `net_arch=[128, 128]`.

Evaluation on a target environment using a checkpoint:

```bash
python test.py
```

Model and `VecNormalize` paths are set at the top of the script — adjust them to the checkpoint you want to evaluate.

To enable UDR, set `udr=True` in `train.py`. Each leg mass is then sampled uniformly in `[0.5, 1.5] × original_mass` at every reset.

### SimOpt + REPS

The SimOpt pipeline assumes a SAC policy already trained on the source environment is available in `SAC/models/`. By default it points to `SAC_CustomHopper-source-v0_seed10_NoUDR_step160000.{zip,pkl}`; update the paths inside `simopt.py` if your checkpoints are named differently.

```bash
cd SAC
python simopt.py --seed 0 --reps 5 --n_points 200 --n_trajs 8 --final_timesteps 200000
```

The script:
1. collects target trajectories using the pre-trained policy,
2. iteratively refines a Gaussian distribution over per-link mass scaling factors by minimising a transition-based discrepancy with REPS,
3. retrains SAC from scratch on the source environment with the recovered scaling factors applied.

The mass scaling factors recovered in the experiments reported in the paper were `[1.114, 1.089, 0.980]`.

## Outputs

Each training run produces, by default:

- a CSV log under `results/` with one row per episode (REINFORCE, AC) or one row per evaluation step (SAC),
- model checkpoints under `models/` (every `--model-every` episodes/steps, plus a `_best` snapshot),
- video recordings under `videos/` (every `--video-every` episodes/steps).

File names encode the run configuration (algorithm, batch size, seed, mode/baseline, UDR flag, step count) so multiple runs can coexist.

## Plotting

`lib/plot_graph.py` exposes `get_series` and `plot_discrepancies` to load several CSV runs (typically across seeds) and plot the mean curve with a min/max or percentile band. The plotting code is meant to be imported from a notebook or a small driver script — there is no standalone CLI.

## Notes

- The environment uses an older Gym API (`env.step` returns 4 values, not 5) and `mujoco-py`. Migration to Gymnasium / `mujoco` is left as future work.
- Trainings on CPU are perfectly feasible; for SAC, GPU offers limited speed-up because the bottleneck is environment stepping.
- Random seeds used in the report: `8`, `15`, `17` for REINFORCE / Actor-Critic; `10`, `15`, `42`, `49` for SAC at various stages.
