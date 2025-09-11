import optuna
from train import main 
import argparse
import numpy as np
import os


def objective(trial):
    
    # Set of proposed values of each parameter
    gamma = trial.suggest_categorical("gamma", [0.99, 0.95])
    baseline = trial.suggest_categorical("baseline", [0.0, 20.0, 50.0, 100.0])
    batch_size = trial.suggest_categorical("batch_size", [50, 10])

    args = argparse.Namespace(
        n_episodes=5000,
        hidden_dim=128,
        gamma=gamma,
        baseline=baseline,
        verbose = True,
        batch_size=batch_size,
        print_every=10000,
        video_every=10000,
        device='cpu',
        seed=1,
        model_every=10000
    )

    # pass the Optuna trial for pruning and enable hyperparameter_search so main returns a metric
    # randomize the search over 3 different seeds and return the average result

    try:
        obj = main(args, trial=trial, hyperparameter_search=True)
    except optuna.exceptions.TrialPruned:
        # print info before re-raising to allow Optuna to account for the prune
        print(f"[Trial {trial.number}] PRUNED — params: {trial.params}")
        raise
    except Exception as e:
        # fallback: print the error and re-raise
        print(f"[Trial {trial.number}] ERROR — params: {trial.params} — exception: {e}")
        raise
    else:
        print(f"[Trial {trial.number}] COMPLETE — params: {trial.params} — value: {obj}")
        return obj

if __name__ == "__main__":
    print("Ciao")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    df = study.trials_dataframe()
    
    out_dir = "optuna"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "optuna_trials_results_def.csv")
    df.to_csv(out_path, index=False)
    print("Saved results to", os.path.abspath(out_path))

    print("Best hyperparameters:", study.best_params)
    print("Best average reward:", study.best_value)

    # Print top-3 completed trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_sorted = sorted(completed, key=lambda t: t.value if t.value is not None else -float("inf"), reverse=True)
    top3 = completed_sorted[:3]

    print("\nTop 3 trials (number, value, params):")
    for t in top3:
        print(f"  Trial {t.number} | value={t.value} | params={t.params}")
