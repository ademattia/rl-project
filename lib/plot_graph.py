import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Iterable, Literal, Mapping, Optional, Tuple, Union



DEFAULT_ALPHA_BAND = 0.20
DEFAULT_LINEWIDTH = 2.0
DEFAULT_FIGSIZE = (7, 5)
DEFAULT_LEGEND_LOC = "best"

def save_fig(fig, filename: str, ext: str = ".png", dpi: int = 300):

    name, t = os.path.splitext(filename)

    if (name in (".png", ".jpg", ".jpeg", ".webp")) and (t == ""):
        name = "figure"
    else:
        t = t.lower()
        if t in (".jpg", ".jpeg", ".webp"):
            ext = t

    try:
        fig.savefig(f"{name}{ext}", dpi=dpi, bbox_inches="tight")
    except:
        print(f"Error while saving figure. Check code or figure object.")


def read_rl_csv(
    path: str,
    *,
    value_col: Literal["Reward", "Episode Steps"] = "Reward",
    stride: int = 1,
    sort_by_episode: bool = True,
) -> np.ndarray:
    if stride < 1:
        raise ValueError("Stride must be >= 1")
    df = pd.read_csv(path)

    if sort_by_episode:
        df = df.sort_values("Episode", kind="mergesort")

    if stride > 1:
        df = df[::stride]
    return df[value_col].to_numpy(dtype=float)


def _ensure_np(a):
    return np.asarray(a, dtype=float)


def aggregate_minmax(
    csv_paths: Iterable[str],
    value_col: Literal["Reward", "Episode Steps"] = "Reward",
    stride: int = 1,
    sort_by_episode: bool = True,
) -> np.ndarray:
    ys = [
        read_rl_csv(
            p, value_col=value_col, stride=stride, sort_by_episode=sort_by_episode
        )
        for p in csv_paths
    ]
    if not ys:
        raise ValueError("No CSV files provided.")

    lengths = list(map(len, ys))
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent lengths after stride: {lengths}.")
    return np.stack(ys)


def get_series(
    groups: Mapping[Union[int, float, str], Iterable[str]],
    value_col: Literal["Reward", "Episode Steps", "Reward/Step"] = "Reward",
    stride: int = 1,
    label_fmt: str = "b={b}",
    sort_by_episode: bool = True,
    percentiles: Optional[Tuple[float, float]] = None,
) -> Dict[str, dict]:

    series: Dict[str, dict] = {}
    for b, paths in groups.items():
        full_reward_runs = []
        full_steps_runs = []
        full_runs = []

        if value_col == "Reward/Step":
            full_reward_runs = []
            for p in paths:
                df = pd.read_csv(p)
                if sort_by_episode:
                    df = df.sort_values("Episode", kind="mergesort")
                full_reward_runs.append(df["Reward"].to_numpy(dtype=float))
                full_steps_runs.append(df["Episode Steps"].to_numpy(dtype=float))

            full_reward_runs = np.stack(full_reward_runs)
            full_steps_runs = np.stack(full_steps_runs)
        elif value_col == "Reward":
            full_runs = [
                read_rl_csv(
                    p, value_col=value_col, stride=1, sort_by_episode=sort_by_episode
                )
                for p in paths
            ]

            if not full_runs:
                raise ValueError("No CSV files provided.")

            lengths = list(map(len, full_runs))
            if len(set(lengths)) != 1:
                raise ValueError(f"Inconsistent lengths: {lengths}.")

            full_runs = np.stack(full_runs)

        if value_col == "Reward/Step":
            n_episodes = full_reward_runs.shape[1]
        elif value_col == "Reward":
            n_episodes = full_runs.shape[1]

        sampled_indices = list(range(stride - 1, n_episodes, stride))

        mean_values = []
        lo_values = []
        hi_values = []

        for idx in sampled_indices:
            start_idx = max(0, idx - stride + 1)

            if value_col == "Reward/Step":
                reward_window = full_reward_runs[:, start_idx : idx + 1]
                steps_window = full_steps_runs[:, start_idx : idx + 1]

                ratio_window = np.divide(
                    reward_window,
                    steps_window,
                    out=np.zeros_like(reward_window),
                    where=steps_window != 0,
                )

                window_mean = ratio_window.mean()

                if percentiles is None:
                    window_min = ratio_window.min()
                    window_max = ratio_window.max()
                    lo_values.append(window_min)
                    hi_values.append(window_max)
                else:
                    p_low, p_high = percentiles
                    window_lo = np.percentile(ratio_window, p_low)
                    window_hi = np.percentile(ratio_window, p_high)
                    lo_values.append(window_lo)
                    hi_values.append(window_hi)
                    
            elif value_col == "Reward":
                # Comportamento normale
                window_data = full_runs[:, start_idx : idx + 1]

                window_mean = window_data.mean(axis=(0, 1))

                if percentiles is None:
                    window_min = window_data.min()
                    window_max = window_data.max()
                    lo_values.append(window_min)
                    hi_values.append(window_max)
                else:
                    p_low, p_high = percentiles
                    window_lo = np.percentile(window_data, p_low)
                    window_hi = np.percentile(window_data, p_high)
                    lo_values.append(window_lo)
                    hi_values.append(window_hi)

            mean_values.append(window_mean)

        mean = np.array(mean_values)
        lo = np.array(lo_values)
        hi = np.array(hi_values)

        series[label_fmt.format(b=b)] = {"mean": mean, "low": lo, "high": hi}

    return series

def plot_discrepancies(
    series: Dict[str, dict],
    title: str,
    xlabel: str,
    ylabel: str,
    value_col: Literal["Reward", "Episode Steps", "Reward/Step"] = "Reward",
    alpha_band: float = DEFAULT_ALPHA_BAND,
    linewidth: float = DEFAULT_LINEWIDTH,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    legend_loc: str = DEFAULT_LEGEND_LOC,
    stride: int = 1,
):
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for label, value in series.items():

        if not all(k in value for k in ("mean", "low", "high")):
            raise ValueError(
                f"'{label}': need 'mean' + 'low' + 'high' for minmax plot."
            )
        y = _ensure_np(value["mean"])
        lo = _ensure_np(value["low"])
        hi = _ensure_np(value["high"])

        x = np.arange(len(y)) * stride
        (line,) = ax.plot(x, y, linewidth=linewidth, label=label)
        if lo is not None and hi is not None:
            ax.fill_between(x, lo, hi, alpha=alpha_band, color=line.get_color())

    if value_col != "Reward/Step":
        ax.axhline(y=500, color="black", linestyle="--", linewidth=1, alpha=0.7)

    ax.legend(loc=legend_loc, frameon=True, framealpha=1.0)
    ax.margins(x=0)
    fig.tight_layout()

    return fig
