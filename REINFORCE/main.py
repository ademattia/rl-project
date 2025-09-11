import subprocess

# File principale REINFORCE (sostituisci con il nome corretto del tuo file)
SCRIPT = "train.py"

# Seeds richiesti
seeds = [15, 8, 17]

# Parametri comuni
n_episodes = 15000
batch_size = 50
video_every = 50000
print_every = 5000
model_every = 30000

# Due baseline da testare
baselines = [0.0, 20.0]

# Loop sugli esperimenti
for baseline in baselines:
    for seed in seeds:
        print(f"\n=== Running baseline={baseline}, seed={seed} ===\n")

        cmd = [
            "python", SCRIPT,
            "--n-episodes", str(n_episodes),
            "--batch-size", str(batch_size),
            "--baseline", str(baseline),
            "--video-every", str(video_every),
            "--print-every", str(print_every),
            "--model-every", str(model_every),
            "--seed", str(seed),
            "--verbose"
        ]

        # Esegui e attendi la fine
        subprocess.run(cmd, check=True)