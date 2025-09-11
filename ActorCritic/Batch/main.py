import subprocess

# File di training (sostituire con il nome corretto del tuo script di training)
SCRIPT = "train.py"

# Parametri fissi
N_EPISODES = 15000
VIDEO_EVERY = 50000
MODEL_EVERY = 5000
PRINT_EVERY = 5000

# Parametri da esplorare
BATCH_SIZES = [1, 10, 50]
SEEDS = [15, 8, 17]
MODES = ["baseline", "TD"]

# Loop su tutte le combinazioni
for batch_size in BATCH_SIZES:
    for mode in MODES:
        for seed in SEEDS:
            print(f"\n=== Running batch_size={batch_size}, mode={mode}, seed={seed} ===\n")

            cmd = [
                "python", SCRIPT,
                "--n-episodes", str(N_EPISODES),
                "--batch-size", str(batch_size),
                "--video-every", str(VIDEO_EVERY),
                "--model-every", str(MODEL_EVERY),
                "--print-every", str(PRINT_EVERY),
                "--seed", str(seed),
                "--mode", mode,
                "--verbose"
            ]

            subprocess.run(cmd, check=True)