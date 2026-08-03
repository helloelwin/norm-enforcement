"""Plot mechanism ROC comparison with one subplot per environment."""

import hashlib
import json
import os
import pickle
from typing import Any, Literal

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import chz

from roc_metrics import aggregate_tpr_fpr, compute_tpr_fpr, has_violation_evaluation
from eval_utils import load_results_with_filters


# Mechanism display names, colors, and markers
MECHANISM_COLORS = {
    "naive": "#7e9e94",
    "checked": "#0ad1ae",
    "checked_backfire": "#8e8aff",
    "checked_repvote": "#ffc61a",
    "checked_escrepvote": "#ff5c5f",
    "checked_cot": "#f2b880",
    "checked_backfire_cot": "#dc8e18",
}

MECHANISM_DISPLAY_NAMES = {
    "naive": "Naive",
    "checked": "Checked",
    "checked_backfire": "Backfire",
    "checked_repvote": "RepVote",
    "checked_escrepvote": "EscRepVote",
    "checked_cot": "CoT",
    "checked_repvote_cot": "RepVote + CoT",
    "checked_backfire_cot": "Backfire + CoT",
}

MECHANISM_MARKERS = {
    "naive": "s",
    "checked": "o",
    "checked_backfire": "o",
    "checked_repvote": "D",
    "checked_escrepvote": "D",
    "checked_cot": "v",
    "checked_repvote_cot": ">",
    "checked_backfire_cot": "^",
}


def configure_matplotlib_for_serif() -> None:
    """Configure matplotlib to use a Times-like serif font (mathtext, no LaTeX)."""
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = 10


@chz.chz
class Config:
    results_dir: str = "results"
    naive_results_dir: str | None = None  # Optional separate dir for naive-mechanism runs
    evaluation_dir: str = "results/plots"

    suffix: str = "mechanism_roc"

    # Multiple environments to compare
    env_names: list[str] = chz.field(default_factory=lambda: ["social", "chatbot", "fishing"])

    # Filters
    agent_model_name: str = "1x1-80b-good-20k-bad-20k"
    agent_context_window_len: int = 5
    mechanism_type: list[str] = chz.field(
        default_factory=lambda: [
            "naive",
            "checked",
            "checked_backfire",
            "checked_repvote",
            "checked_escrepvote",
            # "checked_cot",
            # "checked_repvote_cot",
            # "checked_backfire_cot",
        ]
    )

    max_rounds: int = 15  # Trim runs to this many rounds before computing metrics

    # Truthfulness-specific settings
    include_uncertain: bool = False
    cumulative_denominator: bool = False  # If True, violator/compliant labels use scores up to max_rounds

    show_random_line: bool = True
    show_trajectory: bool = True
    show_auc: bool = True  # Annotate normalized AUC per mechanism in each subplot
    continue_with_random: bool = False

    # Output format: "png" for raster (screen sizing), "pdf" for vector (paper sizing)
    format: Literal["png", "pdf"] = "png"

    use_cache: bool = False
    cache_dir: str = "results/.cache/roc"

    # Figure width in inches (full \textwidth by default)
    # \the\textwidth = 397.48499
    fig_width: float | None = 397.48499 / 72.27 * 1.0

    @chz.init_property
    def save_path(self):
        agent_model_name_short = self.agent_model_name.split("/")[-1]
        envs_str = "-".join(self.env_names)
        ext = self.format
        save_name = f"{self.suffix}_{envs_str}_agent={agent_model_name_short}.{ext}"
        return os.path.join(self.evaluation_dir, save_name)


# =============================================================================
# Trajectory Computation
# =============================================================================


def compute_trajectory(
    runs: list[dict],
    include_uncertain: bool,
    max_rounds: int,
    cumulative_denominator: bool = False,
) -> tuple[list[float], list[float]]:
    """Compute per-round mean (FPR, TPR) trajectory across runs.

    Returns:
        (fprs, tprs) — lists of length max_rounds, one mean value per round.
    """
    if not runs:
        return [], []

    n_rounds = max_rounds
    tpr_matrix = np.zeros((len(runs), n_rounds))
    fpr_matrix = np.zeros((len(runs), n_rounds))

    for i, data in enumerate(runs):
        for r in range(1, n_rounds + 1):
            tpr, fpr = compute_tpr_fpr(data, include_uncertain, max_rounds=r, cumulative_denominator=cumulative_denominator)
            tpr_matrix[i, r - 1] = tpr
            fpr_matrix[i, r - 1] = fpr

    fprs = np.mean(fpr_matrix, axis=0).tolist()
    tprs = np.mean(tpr_matrix, axis=0).tolist()
    return fprs, tprs


def _normalized_auc_ratio(fprs: list[float], tprs: list[float]) -> float:
    """Compute normalized AUC ratio from a (mean) ROC trajectory.

    Trapezoidal area under the curve, normalized by the area under the
    diagonal over the same FPR range:
        A = sum((TPR(t) + TPR(t-1)) / 2 * (FPR(t) - FPR(t-1)))
        ratio = A / (0.5 * FPR(T)^2)
    Returns 0.0 when FPR(T) ~ 0.
    """
    area = 0.0
    for t in range(1, len(fprs)):
        area += (tprs[t] + tprs[t - 1]) / 2.0 * (fprs[t] - fprs[t - 1])
    fpr_final = fprs[-1]
    if fpr_final < 1e-12:
        return 0.0
    return area / (0.5 * fpr_final**2)


def compute_normalized_auc(
    runs: list[dict],
    include_uncertain: bool,
    max_rounds: int,
    cumulative_denominator: bool = False,
) -> float:
    """Normalized AUC of the mean ROC trajectory."""
    if not runs:
        return 0.0
    fprs, tprs = compute_trajectory(runs, include_uncertain, max_rounds, cumulative_denominator)
    if not fprs:
        return 0.0
    return _normalized_auc_ratio([0.0] + fprs, [0.0] + tprs)


# =============================================================================
# Data Loading
# =============================================================================


def _cache_path(config: Config) -> str:
    key_parts = {
        "results_dir": config.results_dir,
        "naive_results_dir": config.naive_results_dir,
        "env_names": list(config.env_names),
        "agent_model_name": config.agent_model_name,
        "agent_context_window_len": config.agent_context_window_len,
        "mechanism_type": list(config.mechanism_type),
        "max_rounds": config.max_rounds,
        "include_uncertain": config.include_uncertain,
        "cumulative_denominator": config.cumulative_denominator,
    }
    digest = hashlib.sha256(json.dumps(key_parts, sort_keys=True).encode()).hexdigest()[:16]
    safe_agent = config.agent_model_name.replace("/", "__")
    return os.path.join(config.cache_dir, f"{safe_agent}_{digest}.pkl")


def load_and_aggregate_all(
    config: Config,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, list[dict]]]]:
    """Load results and aggregate TPR/FPR for each (env, mechanism) pair.

    Returns:
        (all_stats, all_runs) where:
        all_stats: {env_name: {mechanism: {"tpr": {...}, "fpr": {...}, "n_runs": int}}}
        all_runs: {env_name: {mechanism: [list of transcript dicts]}}
    """
    if config.use_cache:
        cache_file = _cache_path(config)
        if os.path.exists(cache_file):
            print(f"\nLoading cached results from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    all_stats: dict[str, dict[str, dict[str, Any]]] = {}
    all_runs: dict[str, dict[str, list[dict]]] = {}

    for env_name in config.env_names:
        print(f"\nLoading results for {env_name}...")
        mechanism_stats: dict[str, dict[str, Any]] = {}
        mechanism_runs: dict[str, list[dict]] = {}

        for mechanism in config.mechanism_type:
            results_dir = config.results_dir
            if mechanism == "naive" and config.naive_results_dir is not None:
                results_dir = config.naive_results_dir
            runs = load_results_with_filters(
                results_dir=results_dir,
                env_name=env_name,
                agent_model_name=config.agent_model_name,
                agent_context_window_len=config.agent_context_window_len,
                mechanism_type=mechanism,
                filter_func=has_violation_evaluation,
            )
            stats = aggregate_tpr_fpr(runs, config.include_uncertain, config.max_rounds, config.cumulative_denominator)
            mechanism_stats[mechanism] = stats
            mechanism_runs[mechanism] = runs

        all_stats[env_name] = mechanism_stats
        all_runs[env_name] = mechanism_runs

    if config.use_cache:
        cache_file = _cache_path(config)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump((all_stats, all_runs), f)
        print(f"  Cached results to {cache_file}")

    return all_stats, all_runs


# =============================================================================
# Layout
# =============================================================================


def finalize_layout_with_bottom_legend_and_suplabels(
    fig: plt.Figure,
    *,
    supylabel: str | None = None,
    supxlabel: str | None = None,
    legend=None,
    extra_bottom_pad: float = 0.02,
    left_pad: float = 0.04,
    bottom_pad: float = 0.04,
    right_pad: float = 0.0,
    w_pad: float = 0.5,
    h_pad: float = 0.5,
    pad: float = 0.8,
    suplabel_fontsize: int = 10,
) -> None:
    """Finalize layout with bottom legend and centered sup-labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    bottom = 0.02
    if legend is not None:
        lb = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        bottom = max(bottom, lb.y1 + extra_bottom_pad)

    fig.tight_layout(rect=[left_pad, bottom, 1 - right_pad, 1], w_pad=w_pad, h_pad=h_pad, pad=pad)

    fig.canvas.draw()

    axes = [ax for ax in fig.axes if ax.get_visible()]
    # Use window extent + figure-coord inverse transform so positions are in the
    # *master* figure's coordinate system even when axes live inside SubFigures
    # (where ax.get_position() is relative to the parent SubFigure).
    bbs = [
        ax.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        for ax in axes
    ]
    x0 = min(bb.x0 for bb in bbs)
    x1 = max(bb.x1 for bb in bbs)
    y0 = min(bb.y0 for bb in bbs)
    y1 = max(bb.y1 for bb in bbs)

    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)

    if legend is not None:
        legend.set_bbox_to_anchor((xc, 0.0), transform=fig.transFigure)

    if supxlabel is not None:
        fig.supxlabel(supxlabel, x=xc, y=max(0.0, y0 - bottom_pad), fontsize=suplabel_fontsize)
    if supylabel is not None:
        fig.supylabel(supylabel, x=max(0.0, x0 - left_pad), y=yc, fontsize=suplabel_fontsize)


# =============================================================================
# Plotting
# =============================================================================


def plot_mechanism_roc_multi_env(
    all_stats: dict[str, dict[str, dict[str, Any]]],
    config: Config,
    all_runs: dict[str, dict[str, list[dict]]] | None = None,
) -> plt.Figure:
    """Create figure with one ROC subplot per environment.

    Args:
        all_stats: {env_name: {mechanism: {"tpr": {...}, "fpr": {...}, "n_runs": int}}}
        config: Config object
        all_runs: {env_name: {mechanism: [runs]}} — needed when show_trajectory is True

    Returns:
        matplotlib Figure
    """
    n_envs = len(all_stats)
    if n_envs == 0:
        raise ValueError("No environment stats to plot")

    # Font sizes per format (pdf bumps everything to 12)
    if config.format == "pdf":
        base_fontsize = 12
        tick_fontsize = 12
    else:
        base_fontsize = 10
        tick_fontsize = 8

    # Figure sizing
    if config.format == "pdf":
        fig_width = config.fig_width if config.fig_width is not None else 5.5
        subplot_height = fig_width / n_envs * 2
        fig_height = subplot_height
    else:
        fig_width = 4 * n_envs
        fig_height = 4

    fig, axes = plt.subplots(1, n_envs, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    # Collect legend handles (one per mechanism, shared across subplots)
    legend_handles = []
    plotted_mechanisms: set[str] = set()

    for idx, (env_name, mechanism_stats) in enumerate(all_stats.items()):
        ax = axes_flat[idx]
        auc_entries: list[tuple[str, str, float]] = []  # (display_name, color, nAUC)

        if config.show_random_line:
            ax.axline((0, 0), slope=1, color="k", linestyle="--", alpha=0.3, linewidth=1.0)

        for mechanism in config.mechanism_type:
            stats = mechanism_stats.get(mechanism)
            if stats is None or stats["n_runs"] == 0:
                continue

            fpr_mean = stats["fpr"]["mean"]
            tpr_mean = stats["tpr"]["mean"]

            fpr_err_lower = fpr_mean - stats["fpr"]["ci_lower"]
            fpr_err_upper = stats["fpr"]["ci_upper"] - fpr_mean
            tpr_err_lower = tpr_mean - stats["tpr"]["ci_lower"]
            tpr_err_upper = stats["tpr"]["ci_upper"] - tpr_mean

            color = MECHANISM_COLORS.get(mechanism, "#333333")
            marker = MECHANISM_MARKERS.get(mechanism, "o")
            display_name = MECHANISM_DISPLAY_NAMES.get(mechanism, mechanism)

            # Draw trajectory line if enabled
            if config.show_trajectory and all_runs is not None:
                runs = all_runs.get(env_name, {}).get(mechanism, [])
                if runs:
                    traj_fprs, traj_tprs = compute_trajectory(
                        runs,
                        config.include_uncertain,
                        config.max_rounds,
                        config.cumulative_denominator,
                    )
                    ax.plot(traj_fprs, traj_tprs, color=color, linewidth=1, alpha=0.9)

            # Draw line from endpoint to (1, 1) if enabled
            if config.continue_with_random:
                ax.plot(
                    [fpr_mean, 1.0],
                    [tpr_mean, 1.0],
                    color=color,
                    linewidth=1,
                    alpha=0.4,
                    linestyle="-",
                )

            ax.errorbar(
                fpr_mean,
                tpr_mean,
                xerr=[[fpr_err_lower], [fpr_err_upper]],
                yerr=[[tpr_err_lower], [tpr_err_upper]],
                fmt="none",
                ecolor=color,
                capsize=3,
                capthick=1.5,
                elinewidth=1.5,
                alpha=0.2,
            )
            ax.scatter(
                fpr_mean,
                tpr_mean,
                marker=marker,
                color=color,
                s=20 if config.format != "pdf" else 28,
                zorder=5,
            )

            # Collect normalized AUC for this (env, mechanism) for annotation
            if config.show_auc and all_runs is not None:
                runs = all_runs.get(env_name, {}).get(mechanism, [])
                if runs:
                    auc = compute_normalized_auc(
                        runs,
                        config.include_uncertain,
                        config.max_rounds,
                        config.cumulative_denominator,
                    )
                    auc_entries.append((display_name, color, auc))

            # Build legend handle once per mechanism
            if mechanism not in plotted_mechanisms:
                plotted_mechanisms.add(mechanism)
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=marker,
                        color=color,
                        linestyle="None",
                        markersize=6 if config.format != "pdf" else 10,
                        label=display_name,
                    )
                )

        # Annotate normalized AUC per mechanism in the (empty) bottom-right corner
        if auc_entries:
            auc_fontsize = tick_fontsize - 1
            line_h = 0.062
            ax.text(
                0.96,
                0.04 + line_h * len(auc_entries),
                "nAUC",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=auc_fontsize,
                color="#333333",
                zorder=6,
            )
            for j, (name, entry_color, auc) in enumerate(reversed(auc_entries)):
                ax.text(
                    0.96,
                    0.04 + line_h * j,
                    f"{name}: {auc:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=auc_fontsize,
                    color=entry_color,
                    zorder=6,
                )

        ax.set_box_aspect(1 if config.format == "pdf" else 1.0)
        ax.margins(0.15)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

        # Format ticks: 1 decimal place, drop leading zero (e.g. 0.3 → .3)
        def _fmt_tick(x, _pos):
            if x == int(x):
                return f"{int(x)}"
            return f"{x:.1f}".replace("0.", ".")

        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_tick))
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_tick))
        ax.set_title(env_name.capitalize(), fontsize=base_fontsize)

    # Hide unused axes
    for idx in range(n_envs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared bottom legend
    leg = None
    if legend_handles:
        leg = fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(legend_handles),
            frameon=False,
            borderpad=0.0,
            handletextpad=0.4,
            columnspacing=0.8,
            handlelength=1.0,
            borderaxespad=0.5,
        )

    finalize_layout_with_bottom_legend_and_suplabels(
        fig,
        supxlabel="False Positive Rate ($\\downarrow$)" if config.format != "pdf" else "",
        supylabel="True Positive Rate ($\\uparrow$)" if config.format != "pdf" else "",
        legend=leg,
        extra_bottom_pad=0.15,
        left_pad=0.07,
        bottom_pad=0.16,
        suplabel_fontsize=base_fontsize,
    )

    return fig


# =============================================================================
# Main
# =============================================================================


def main(config: Config):
    """Entry point for multi-environment mechanism ROC plot."""
    configure_matplotlib_for_serif()

    os.makedirs(config.evaluation_dir, exist_ok=True)

    # Load and aggregate
    all_stats, all_runs = load_and_aggregate_all(config)

    # Check for data
    total_runs = sum(stats["n_runs"] for env_stats in all_stats.values() for stats in env_stats.values())
    if total_runs == 0:
        print("No matching results found with completed violation evaluation.")
        return

    # Print summary
    print("\n" + "=" * 60)
    print("MECHANISM ROC SUMMARY")
    print("=" * 60)
    for env_name, mechanism_stats in all_stats.items():
        print(f"\n--- {env_name} ---")
        for mechanism, stats in sorted(mechanism_stats.items()):
            if stats["n_runs"] == 0:
                continue
            print(
                f"  {mechanism} (n={stats['n_runs']}): "
                f"TPR={stats['tpr']['mean']:.3f} [{stats['tpr']['ci_lower']:.3f}, {stats['tpr']['ci_upper']:.3f}], "
                f"FPR={stats['fpr']['mean']:.3f} [{stats['fpr']['ci_lower']:.3f}, {stats['fpr']['ci_upper']:.3f}]"
            )

    # Print normalized AUC per mechanism (same values as the per-subplot annotations)
    print("\n" + "=" * 60)
    print("NORMALIZED AUC BY MECHANISM")
    print("=" * 60)
    for env_name, mechanism_stats in all_stats.items():
        print(f"\n--- {env_name} ---")
        for mechanism in config.mechanism_type:
            stats = mechanism_stats.get(mechanism)
            if stats is None or stats["n_runs"] == 0:
                continue
            runs = all_runs.get(env_name, {}).get(mechanism, [])
            auc = compute_normalized_auc(
                runs, config.include_uncertain, config.max_rounds, config.cumulative_denominator
            )
            display_name = MECHANISM_DISPLAY_NAMES.get(mechanism, mechanism)
            print(f"  {display_name} ({mechanism}, n={stats['n_runs']}): AUC={auc:.3f}")

    # Plot
    fig = plot_mechanism_roc_multi_env(all_stats, config, all_runs)

    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    if config.format == "pdf":
        w_in, h_in = fig.get_size_inches()
        print(f"Saving PDF at figsize=({w_in:.3f}, {h_in:.3f}) inches")
        # Override savefig.bbox rcParam (might be "tight" from matplotlibrc).
        with mpl.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0}):
            fig.savefig(config.save_path, bbox_inches=None, pad_inches=0)
    else:
        fig.savefig(config.save_path, dpi=150, bbox_inches="tight")

    print(f"\nSaved plot to {config.save_path}")

    plt.close(fig)

    # Print run lengths per mechanism per environment
    print("\n" + "=" * 60)
    print(f"RUN LENGTHS BY MECHANISM (max_rounds={config.max_rounds})")
    print("=" * 60)
    for env_name, mechanism_runs in all_runs.items():
        print(f"\n--- {env_name} ---")
        for mechanism, runs in sorted(mechanism_runs.items()):
            if not runs:
                continue
            lengths = [len(d.get("results", [])) for d in runs]
            trimmed = [min(l, config.max_rounds) for l in lengths]
            print(f"  {mechanism} ({len(runs)} runs):")
            print(f"    Original lengths: {sorted(lengths)}")
            if any(l > config.max_rounds for l in lengths):
                print(f"    Trimmed lengths:  {sorted(trimmed)}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
