"""Plot accurate vs false report counts by model, with one subplot per environment."""

import os
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import chz

from report_metrics import (
    aggregate_metrics,
    has_violation_evaluation,
)
from eval_utils import load_results_with_filters, trim_runs


COLOR_GREEN = "#84a98c"
COLOR_RED = "#edafb8"


@chz.chz
class Config:
    results_dir: str = "results"
    evaluation_dir: str = "results/plots"

    suffix: str = "report_counts"

    # Multiple environments to compare
    env_names: list[str] = chz.field(default_factory=lambda: ["social", "chatbot", "fishing"])

    # Filters (applied to all environments)
    agent_model_name: str = "1x1-80b-good-20k-bad-20k"
    agent_context_window_len: int = 5
    mechanism_type: str = "naive"

    max_rounds: int = 15  # Trim runs to this many rounds before computing metrics

    # Truthfulness-specific settings
    include_uncertain: bool = False
    per_agent_round: bool = True # If True, normalize report counts by total agent-rounds

    # Output format: "png" for raster (screen sizing), "pdf" for vector (paper sizing)
    format: Literal["png", "pdf"] = "png"

    # Figure size in inches, used for the compact "pdf" (paper) sizing
    fig_width: float | None = 397.48499 / 72.27 * 0.52
    fig_height: float = 1.8

    left_pad: float = 0.14

    ymax: float | None = None

    @chz.init_property
    def save_path(self):
        agent_model_name_short = self.agent_model_name.split("/")[-1]
        envs_str = "-".join(self.env_names)
        ext = self.format
        per_ar_str = "_per_agent_round" if self.per_agent_round else ""
        save_name = (
            f"{self.suffix}_{envs_str}_agent={agent_model_name_short}"
            f"_mechanism={self.mechanism_type}{per_ar_str}.{ext}"
        )
        return os.path.join(self.evaluation_dir, save_name)


def load_results_for_env(config: Config, env_name: str) -> list[dict]:
    """Load and filter results for a specific environment."""
    return load_results_with_filters(
        results_dir=config.results_dir,
        env_name=env_name,
        agent_model_name=config.agent_model_name,
        agent_context_window_len=config.agent_context_window_len,
        mechanism_type=config.mechanism_type,
        filter_func=has_violation_evaluation,
    )


def finalize_layout_with_bottom_legend_and_suplabels(
    fig: plt.Figure,
    *,
    supylabel: str | None = None,
    supxlabel: str | None = None,
    legend=None,  # Legend object or None
    extra_bottom_pad: float = 0.02,  # extra space above legend
    left_pad: float = 0.04,  # space left of subplot block for supylabel
    bottom_pad: float = 0.04,  # space below subplot block for supxlabel
) -> None:
    """
    1) Draw to finalize artist sizes
    2) Compute how much bottom space the legend needs (if any)
    3) tight_layout with an automatically chosen rect bottom
    4) Re-center supxlabel/supylabel on the *actual* subplot block
    """
    # Ensure we have a renderer
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # How much bottom to reserve?
    bottom = 0.02  # baseline
    if legend is not None:
        # Legend bbox in figure fraction coords
        lb = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        bottom = max(bottom, lb.y1 + extra_bottom_pad)

    # Apply tight_layout, reserving left strip for supylabel and bottom strip for legend
    fig.tight_layout(rect=[left_pad, bottom, 1, 1], w_pad=0, h_pad=0.5, pad=0.8)

    # Draw again so axes positions reflect tight_layout
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Redistribute subplots so gaps adapt to actual tick-label widths
    visible_axes = sorted(
        [ax for ax in fig.axes if ax.get_visible()],
        key=lambda a: a.get_position().x0,
    )
    if len(visible_axes) > 1:
        positions = [ax.get_position() for ax in visible_axes]
        tight_bbs = [ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()) for ax in visible_axes]
        # Overhang of tick labels beyond the axes box on each side
        left_oh = [p.x0 - t.x0 for p, t in zip(positions, tight_bbs)]
        right_oh = [t.x1 - p.x1 for p, t in zip(positions, tight_bbs)]
        plot_w = positions[0].width  # equal from tight_layout
        # Total tight width (axes + their label overhangs)
        total_tight = sum(lo + plot_w + ro for lo, ro in zip(left_oh, right_oh))
        avail = 1.0 - left_pad
        n = len(visible_axes)
        gap = max(0, (avail - total_tight) / (n - 1)) * 0.4 if n > 1 else 0
        # Lay out left-to-right
        cursor = left_pad  # left edge of first tight bbox
        for i, ax in enumerate(visible_axes):
            ax.set_position([cursor + left_oh[i], positions[i].y0, plot_w, positions[i].height])
            cursor += left_oh[i] + plot_w + right_oh[i] + gap
        fig.canvas.draw()

    # Union bbox of visible axes (in figure fraction coords)
    axes = [ax for ax in fig.axes if ax.get_visible()]
    bbs = [ax.get_position() for ax in axes]
    x0 = min(bb.x0 for bb in bbs)
    x1 = max(bb.x1 for bb in bbs)
    y0 = min(bb.y0 for bb in bbs)
    y1 = max(bb.y1 for bb in bbs)

    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)

    # Re-center legend to subplot block (ignoring supylabel width)
    if legend is not None:
        legend.set_bbox_to_anchor((xc, 0.0), transform=fig.transFigure)

    # Place sups relative to the *subplot block*, not the whole figure
    if supxlabel is not None:
        fig.supxlabel(supxlabel, x=xc, y=max(0.0, y0 - bottom_pad), fontsize=10)
    if supylabel is not None:
        fig.supylabel(supylabel, x=max(0.0, x0 - left_pad), y=yc, fontsize=10)


def plot_report_counts_multi_env(
    env_stats: dict[str, dict[str, dict[str, Any]]],
    config: Config,
) -> plt.Figure:
    """Create figure with one subplot per environment showing accurate/false reports.

    Args:
        env_stats: {env_name: {model_name: rate_stats}}
        config: Config object

    Returns:
        matplotlib Figure
    """
    n_envs = len(env_stats)
    if n_envs == 0:
        raise ValueError("No environment stats to plot")

    n_cols = n_envs

    # Figure sizing
    if config.format == "pdf":
        subplot_width = 1.6
        if config.fig_width is not None:
            subplot_width = config.fig_width / n_cols
        fig_width = subplot_width * n_cols
    else:
        fig_width = 4 * n_cols
    fig_height = config.fig_height

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    color_accurate = COLOR_GREEN
    color_false = COLOR_RED
    handles, labels = None, None

    for idx, (env_name, rate_stats) in enumerate(env_stats.items()):
        ax = axes_flat[idx]

        if not rate_stats:
            ax.set_title(f"{env_name} (no data)")
            ax.set_visible(False)
            continue

        model_names = sorted(rate_stats.keys())
        x = np.arange(len(model_names))
        width = 0.45

        accurate_means = [rate_stats[m]["avg_accurate"]["mean"] for m in model_names]
        accurate_errs = [
            [rate_stats[m]["avg_accurate"]["mean"] - rate_stats[m]["avg_accurate"]["ci_lower"] for m in model_names],
            [rate_stats[m]["avg_accurate"]["ci_upper"] - rate_stats[m]["avg_accurate"]["mean"] for m in model_names],
        ]

        false_means = [rate_stats[m]["avg_false"]["mean"] for m in model_names]
        false_errs = [
            [rate_stats[m]["avg_false"]["mean"] - rate_stats[m]["avg_false"]["ci_lower"] for m in model_names],
            [rate_stats[m]["avg_false"]["ci_upper"] - rate_stats[m]["avg_false"]["mean"] for m in model_names],
        ]

        h1 = ax.bar(
            x - width / 2,
            accurate_means,
            width,
            yerr=accurate_errs,
            label="Accurate Reports",
            color=color_accurate,
            ecolor=tuple(c * 0.8 for c in mcolors.to_rgb(color_accurate)),
            capsize=2,
            error_kw={"elinewidth": 1.5, "alpha": 0.9},
        )
        h2 = ax.bar(
            x + width / 2,
            false_means,
            width,
            yerr=false_errs,
            label="False Reports",
            color=color_false,
            ecolor=tuple(c * 0.8 for c in mcolors.to_rgb(color_false)),
            capsize=2,
            error_kw={"elinewidth": 1.5, "alpha": 0.9},
        )

        if handles is None:
            handles = [h1, h2]
            labels = ["Accurate Reports", "False Reports"]

        ax.set_title(f"{env_name.capitalize()}", fontsize=10)
        ax.set_xticks(x)
        label_fontsize = 8
        ax.set_xticklabels(
            [
                ("$\\pi_{\\mathrm{good}}$" if "good" in m else "$\\pi_{\\mathrm{bad}}$")
                for m in model_names
            ],
            fontsize=label_fontsize,
        )
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 4, 5, 10]))
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".").replace("0.", "."))
        )
        if config.ymax is not None:
            ax.set_ylim(0, config.ymax)
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused axes
    for idx in range(n_envs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Create legend first (so we can measure it)
    leg = None
    if handles and labels:
        leg = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=2,
            fontsize=8,
            # no border / no background
            frameon=False,
            # tighter packing
            borderpad=0.0,  # (mostly irrelevant if frameon=False, but harmless)
            handletextpad=0.4,  # space between handle and text
            labelspacing=0.2,  # vertical space between entries (if multiple rows)
            columnspacing=0.8,  # space between columns
            handlelength=1.0,  # length of the legend line/box
            borderaxespad=0.5,  # padding between legend and axes/anchor
        )

    # One clean call to do:
    # - auto reserve bottom space based on legend height
    # - tight_layout
    # - recenter supxlabel/supylabel to the actual axes block
    finalize_layout_with_bottom_legend_and_suplabels(
        fig,
        supxlabel="Agent Model",
        supylabel="Count",
        legend=leg,
        extra_bottom_pad=0.05,  # tweak if you want more/less air above legend
        left_pad=config.left_pad,  # tweak if you want supylabel closer/farther
        bottom_pad=0.21,  # tweak if you want supxlabel closer/farther
    )

    return fig


def main(config: Config):
    """Entry point for multi-environment report counts plot."""
    os.makedirs(config.evaluation_dir, exist_ok=True)

    # Collect stats for each environment
    env_stats: dict[str, dict[str, dict[str, Any]]] = {}

    for env_name in config.env_names:
        print(f"\nLoading results for {env_name}...")
        runs = load_results_for_env(config, env_name)

        if not runs:
            print(f"  No matching results found for {env_name}")
            env_stats[env_name] = {}
            continue

        runs = trim_runs(runs, config.max_rounds)
        rate_stats = aggregate_metrics(runs, config.include_uncertain, config.per_agent_round)
        env_stats[env_name] = rate_stats

        # Print summary
        print(f"  Found {len(runs)} runs")
        for model_name, stats in sorted(rate_stats.items()):
            print(
                f"    {model_name}: "
                f"accurate={stats['avg_accurate']['mean']:.4f}, "
                f"false={stats['avg_false']['mean']:.4f}"
            )

    # Check if we have any data
    if not any(env_stats.values()):
        print("\nNo data found for any environment.")
        return

    # Plot
    fig = plot_report_counts_multi_env(env_stats, config)

    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    if config.format == "pdf":
        fig.savefig(config.save_path, bbox_inches="tight", transparent=True)
    else:
        fig.savefig(config.save_path, dpi=150, bbox_inches="tight", transparent=True)

    print(f"\nSaved plot to {config.save_path}")

    plt.close(fig)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
