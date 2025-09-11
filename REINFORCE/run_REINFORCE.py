import subprocess
import argparse
import json
import datetime
import os
import time
from collections import deque
from threading import Thread
import sys
import psutil


def parse_args():
    ap = argparse.ArgumentParser(
        description="Parallel orchestrator for train.py (REINFORCE)"
    )
    ap.add_argument("--python", default="python")
    ap.add_argument("--script", default="train.py")
    ap.add_argument("--n-episodes", type=int, default=15000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[8, 15, 17])
    ap.add_argument("--baselines", nargs="+", type=float, default=[0.0, 20.0, 50.0])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 10, 50])

    # Logging and saving settings
    ap.add_argument("--video-every", type=int, default=30000)
    ap.add_argument("--print-every", type=int, default=1000)
    ap.add_argument("--model-every", type=int, default=5000)
    ap.add_argument("--status-file", default="run_status_REINFORCE.json")
    ap.add_argument("--logs-dir", default="logs")

    # Parallelism settings
    ap.add_argument("--device", default="cpu", type=str, help="device [cpu, cuda]")
    ap.add_argument("--max-procs", type=int, default=8)
    ap.add_argument("--gpu-slots", type=int, default=1)
    ap.add_argument("--cpu-fallback", action="store_true")
    ap.add_argument("--pin-cpus", action="store_true")
    ap.add_argument(
        "--cpus-per-proc",
        type=int,
        default=1,
        help="How many logical CPUs to grant each process (1 or 2 recommended)",
    )

    # Logging behavior
    ap.add_argument(
        "--log-mode",
        choices=["file", "inherit", "tee"],
        default="tee",
        help="file: only to log file; inherit: only to console; tee: both",
    )
    ap.add_argument(
        "--unbuffered",
        action="store_true",
        default=True,
        help="Run python -u and PYTHONUNBUFFERED=1",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Test run without actual training - verify model/CSV paths and configurations",
    )

    return ap.parse_args()


# ——————————————————————————————————————————————————————————————


def build_cmd(
    python_exe,
    script,
    n_episodes,
    batch_size,
    baseline,
    video_every,
    print_every,
    model_every,
    seed,
    device,
    extra_env=None,
    cpu_list=None,
    unbuffered=True,
    dry_run=False,
):
    """
    Build a command for one training job.
    Optionally pin CPUs with taskset and enable unbuffered Python I/O.
    """
    base_cmd = [python_exe]
    if unbuffered:
        base_cmd.append("-u")  # unbuffered stdout/stderr

    base_cmd += [
        script,
        "--n-episodes",
        str(n_episodes),
        "--batch-size",
        str(batch_size),
        "--baseline",
        str(baseline),
        "--video-every",
        str(video_every),
        "--print-every",
        str(print_every),
        "--model-every",
        str(model_every),
        "--seed",
        str(seed),
        "--verbose",
        "--device",
        device,
    ]

    if dry_run:
        base_cmd.append("--dry-run")

    # Optional CPU affinity (Linux)
    prefix = []
    if cpu_list:
        cpu_str = ",".join(str(c) for c in cpu_list)
        prefix = ["taskset", "-c", cpu_str]

    full_cmd = prefix + base_cmd

    # Per-process environment
    env = os.environ.copy()
    env.setdefault(
        "PYTHONUNBUFFERED", "1" if unbuffered else env.get("PYTHONUNBUFFERED", "0")
    )
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    if extra_env:
        env.update(extra_env)

    return full_cmd, env


def cpu_topology_pairs():
    base = "/sys/devices/system/cpu"
    pairs = []
    try:
        cpus = sorted(
            [d for d in os.listdir(base) if d.startswith("cpu") and d[3:].isdigit()],
            key=lambda x: int(x[3:]),
        )
        seen = set()
        for c in cpus:
            idx = int(c[3:])
            if idx in seen:
                continue
            sib_path = os.path.join(base, c, "topology", "thread_siblings_list")
            if os.path.exists(sib_path):
                with open(sib_path) as f:
                    txt = f.read().strip()
                sibs = []
                for part in txt.split(","):
                    if "-" in part:
                        a, b = part.split("-")
                        sibs.extend(list(range(int(a), int(b) + 1)))
                    else:
                        sibs.append(int(part))
                sibs = sorted(set(sibs))
                for s in sibs:
                    seen.add(s)
                pairs.append(sibs)
            else:
                pairs.append([idx])
                seen.add(idx)
        return pairs
    except Exception:
        # Fallback: naive adjacent pairing
        n = os.cpu_count() or 12
        return [[i] for i in range(n)]


def map_worker_to_cpus(worker_index, cpus_per_proc=1):
    pairs = cpu_topology_pairs()
    core_idx = worker_index % len(pairs)
    sibs = pairs[core_idx]
    if cpus_per_proc >= 2 and len(sibs) >= 2:
        return [sibs[0], sibs[1]]  
    return [sibs[0]]


def stream_tee(proc, log_path, prefix):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", buffering=1) as f: 
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            # Write to console with a prefix to identify the run
            sys.stdout.write(f"{prefix} {line}")
            sys.stdout.flush()
            # Write to file
            f.write(line)
            f.flush()


def main():
    args = parse_args()

    runs = [
        {"batch_size": b, "baseline": bl, "seed": s}
        for b in args.batch_sizes
        for bl in args.baselines
        for s in args.seeds
    ]
    total_runs = len(runs)
    start_time = datetime.datetime.now()

    status = {
        "algorithm": "REINFORCE",
        "start_time": start_time.isoformat(),
        "total_runs": total_runs,
        "completed_runs": 0,
        "failed_runs": 0,
        "running": 0,
        "runs": [],
    }

    pending = deque(runs)
    running = {}
    gpu_in_use = 0

    try:
        total_logical_cpus = psutil.cpu_count(logical=True) or (os.cpu_count() or 12)
    except Exception:
        total_logical_cpus = os.cpu_count() or 12

    def flush_status():
        with open(args.status_file, "w") as f:
            json.dump(status, f, indent=2)

    print(
        f"[orchestrator] {total_runs} runs; max-procs={args.max_procs}, gpu-slots={args.gpu_slots}, cpu-fallback={args.cpu_fallback}"
    )
    print(f"[orchestrator] logical CPUs detected: {total_logical_cpus}")
    print(
        f"[orchestrator]  LOGS: {args.logs_dir}  |  STATUS: {args.status_file}  |  LOG-MODE: {args.log_mode}"
    )

    worker_index = 0

    try:
        while pending or running:
            # Launch window
            while pending and len(running) < args.max_procs:
                job = pending[0]
                needs_gpu = True
                device = "cuda"
                if gpu_in_use >= args.gpu_slots:
                    if args.cpu_fallback:
                        needs_gpu = False
                        device = "cpu"
                    else:
                        break

                batch_size = job["batch_size"]
                baseline = job["baseline"]
                seed = job["seed"]
                log_name = f"batch_{batch_size}_baseline_{baseline}_seed_{seed}.log"
                log_path = os.path.join(args.logs_dir, log_name)

                cpu_list = None
                if args.pin_cpus:
                    cpu_list = map_worker_to_cpus(worker_index, args.cpus_per_proc)
                    worker_index += 1

                cmd, env = build_cmd(
                    python_exe=args.python,
                    script=args.script,
                    n_episodes=args.n_episodes,
                    batch_size=batch_size,
                    baseline=baseline,
                    video_every=args.video_every,
                    print_every=args.print_every,
                    model_every=args.model_every,
                    seed=seed,
                    device=device,
                    extra_env=None,
                    cpu_list=cpu_list,
                    unbuffered=args.unbuffered,
                    dry_run=args.dry_run,
                )

                if args.log_mode == "inherit":
                    proc = subprocess.Popen(cmd, env=env, close_fds=True)
                    fh = None
                elif args.log_mode == "file":
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    fh = open(log_path, "w", buffering=1)
                    proc = subprocess.Popen(
                        cmd,
                        env=env,
                        close_fds=True,
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                    )
                else:  
                    proc = subprocess.Popen(
                        cmd,
                        env=env,
                        close_fds=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    fh = None

                pid = proc.pid
                print(
                    f"\n[launch] BASELINE={baseline} SEED={seed} DEVICE={device}"
                    + (f" CPU={cpu_list}" if cpu_list else "")
                    + f" | LOG={log_path}"
                    + f" | PID={pid}"
                    + (" | DRY-RUN" if args.dry_run else "")
                    + "\n"
                )

                meta = {
                    "batch_size": batch_size,
                    "baseline": baseline,
                    "seed": seed,
                    "device": device,
                    "pid": pid,
                    "start_time": datetime.datetime.now().isoformat(),
                    "status": "running",
                    "log": log_path,
                    "cmd": cmd,
                }
                status["runs"].append(meta)
                status["running"] = len(running) + 1
                if needs_gpu:
                    gpu_in_use += 1

                tee_thread = None
                if args.log_mode == "tee":
                    prefix = f"[run b={baseline} s={seed}]"
                    tee_thread = Thread(
                        target=stream_tee, args=(proc, log_path, prefix), daemon=True
                    )
                    tee_thread.start()

                if args.log_mode == "file":
                    running[pid] = (proc, meta, None, fh)
                else:
                    running[pid] = (proc, meta, tee_thread, None)

                pending.popleft()
                flush_status()

            # Poll children
            if running:
                time.sleep(0.5)
                for pid in list(running.keys()):
                    proc, meta, tee_thread, fh = running[pid]
                    ret = proc.poll()
                    if ret is None:
                        continue

                    # Close file handle if needed
                    if fh is not None:
                        try:
                            fh.close()
                        except Exception:
                            pass

                    meta["end_time"] = datetime.datetime.now().isoformat()
                    if ret == 0:
                        meta["status"] = "completed"
                        print(
                            f"[done] pid={pid} baseline={meta['baseline']} seed={meta['seed']} device={meta['device']} OK"
                        )
                        status["completed_runs"] += 1
                    else:
                        meta["status"] = "failed"
                        meta["returncode"] = ret
                        print(
                            f"[fail] pid={pid} baseline={meta['baseline']} seed={meta['seed']} device={meta['device']} rc={ret}"
                        )
                        status["failed_runs"] += 1

                    if meta["device"] == "cuda":
                        gpu_in_use = max(0, gpu_in_use - 1)

                    del running[pid]
                    status["running"] = len(running)
                    flush_status()
            else:
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[orchestrator] Ctrl-C, terminating children...")
        for pid, (proc, meta, tee_thread, fh) in list(running.items()):
            try:
                proc.terminate()
            except Exception:
                pass
        for pid, (proc, meta, tee_thread, fh) in list(running.items()):
            try:
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    # Finalize
    end_time = datetime.datetime.now()
    status["end_time"] = end_time.isoformat()
    status["total_duration"] = str(
        end_time - datetime.datetime.fromisoformat(status["start_time"])
    )
    with open(args.status_file, "w") as f:
        json.dump(status, f, indent=2)

    print("\n=== Execution Summary ===")
    print(f"Total runs:   {status['total_runs']}")
    print(f"Completed:    {status['completed_runs']}")
    print(f"Failed:       {status['failed_runs']}")
    print(f"Still running:{status.get('running', 0)}")
    print(f"Status saved to: {args.status_file}")


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    try:
        main()
    except KeyboardInterrupt:
        print(f"Training interrupted by user.")
