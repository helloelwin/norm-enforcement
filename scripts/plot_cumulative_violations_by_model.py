"""Plot cumulative violations over time for multiple environments and models in a single plot."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import chz

from report_metrics import aggregate_cumulative_violations, has_violation_evaluation
from eval_utils import load_results_with_filters, trim_runs


@chz.chz
class Config:
    results_dir: str = "results"
    evaluation_dir: str = "results/plots"

    suffix: str = "cumulative_violations"

    # Multiple environments to compare
    env_names: list[str] = chz.field(default_factory=lambda: ["social", "chatbot", "fishing"])

    # Filters (applied to all environments)
    agent_model_name: str = "1x1-80b-good-20k-bad-20k"
    agent_context_window_len: int = 5
    mechanism_type: str = "naive"

    max_rounds: int = 15  # Trim runs to this many rounds before computing metrics

    # Violation threshold settings
    include_uncertain: bool = False

    # Output format: "png" for raster (screen sizing), "pdf" for vector (paper sizing)
    format: Literal["png", "pdf"] = "png"

    # Figure size in inches, used for the compact "pdf" (paper) sizing
    fig_width: float | None = 397.48499 / 72.27 * 0.28
    fig_height: float = 1.8

    @chz.init_property
    def save_path(self):
        agent_model_name_short = self.agent_model_name.split("/")[-1]
        envs_str = "-".join(self.env_names)
        ext = self.format
        save_name = (
            f"{self.suffix}_{envs_str}_agent={agent_model_name_short}"
            f"_mechanism={self.mechanism_type}.{ext}"
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
    legend=None,
    extra_bottom_pad: float = 0.02,
    left_pad: float = 0.04,
    bottom_pad: float = 0.04,
) -> None:
    """Finalize layout with bottom legend and centered sup-labels.

    1) Draw to finalize artist sizes
    2) Compute how much bottom space the legend needs (if any)
    3) tight_layout with an automatically chosen rect bottom
    4) Re-center supxlabel/supylabel on the *actual* subplot block
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    bottom = 0.02
    if legend is not None:
        lb = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        bottom = max(bottom, lb.y1 + extra_bottom_pad)

    fig.tight_layout(rect=[0, bottom, 1, 1], pad=0.8)

    # Draw again so axes positions reflect tight_layout
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
        legend.set_bbox_to_anchor((xc * 1.0, 0.0), transform=fig.transFigure)

    # Place sups relative to the *subplot block*, not the whole figure
    if supxlabel is not None:
        fig.supxlabel(supxlabel, x=xc, y=max(0.0, y0 - bottom_pad), fontsize=10)
    if supylabel is not None:
        fig.supylabel(supylabel, x=max(0.0, x0 - left_pad), y=yc, fontsize=10)


def plot_cumulative_violations_multi_env(
    env_stats: dict[str, dict[str, dict[str, np.ndarray]]],
    config: Config,
) -> plt.Figure:
    """Create figure with cumulative violations for all environments and models.

    Args:
        env_stats: {env_name: {model_name: {"mean", "ci_lower", "ci_upper"}}}
        config: Config object

    Returns:
        matplotlib Figure
    """
    if not env_stats:
        raise ValueError("No environment stats to plot")

    # Figure sizing
    if config.format == "pdf":
        fig_width = config.fig_width if config.fig_width is not None else 3.5
        fig_height = config.fig_height
    else:
        fig_width = 6
        fig_height = 4.5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    env_color_map = {
        "social": "#edafb8",
        "chatbot": "#7faddc",
        "fishing": "#84a98c",
    }
    linestyles = ["-", "--", ":", "-."]

    # Build mappings for colors (environments) and linestyles (models)
    env_names = list(env_stats.keys())
    env_colors = {env: env_color_map.get(env, "#333333") for env in env_names}

    # Collect all unique model names across environments
    all_models = set()
    for env_data in env_stats.values():
        all_models.update(env_data.keys())
    sorted_models = sorted(all_models)
    model_linestyles = {model: linestyles[i % len(linestyles)] for i, model in enumerate(sorted_models)}

    # Plot each (env, model) combination
    for env_name in env_names:
        env_data = env_stats[env_name]
        color = env_colors[env_name]

        for model_name in sorted_models:
            if model_name not in env_data:
                continue

            stats = env_data[model_name]
            mean = stats["mean"]
            ci_lower = stats["ci_lower"]
            ci_upper = stats["ci_upper"]

            num_rounds = len(mean)
            if num_rounds == 0:
                continue

            x_rounds = np.arange(1, num_rounds + 1)
            linestyle = model_linestyles[model_name]

            # Plot CI band
            ax.fill_between(x_rounds, ci_lower, ci_upper, color=color, alpha=0.15)

            # Plot mean line
            ax.plot(x_rounds, mean, color=color, linestyle=linestyle, linewidth=1.5)

    # Configure axes (no xlabel here - will use supxlabel for alignment consistency)
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 15)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}" if v % 1 == 0 else f"{v:.1f}".lstrip("0")))
    ax.grid(True, alpha=0.3)

    ax.set_title("Norm Violations", fontsize=10)

    # Color legend (inside plot) - for environments
    env_handles = [
        Line2D([0], [0], color=env_colors[env], linestyle="-", linewidth=2.0, label=env.capitalize())
        for env in env_names
    ]
    ax.legend(handles=env_handles, loc="lower left", fontsize=7, handlelength=0.8, framealpha=0.5)

    # Line style legend (bottom of figure) - for models
    model_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=model_linestyles[m],
            linewidth=1.5,
            label=("$\\pi_{\\mathrm{good}}$" if "good" in m else ("$\\pi_{\\mathrm{bad}}$" if "bad" in m else m)),
        )
        for m in sorted_models
    ]
    fig_legend = fig.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(sorted_models),
        fontsize=8,
        frameon=False,
        borderpad=0.0,
        handletextpad=0.4,
        columnspacing=0.8,
        handlelength=1.5,
        borderaxespad=0.5,
    )

    finalize_layout_with_bottom_legend_and_suplabels(
        fig,
        supxlabel="Round",
        legend=fig_legend,
        extra_bottom_pad=0.05,  # Increased to add more space above legend
        bottom_pad=0.195,  # Adjusted to align with other plot's supxlabel
    )

    return fig


def main(config: Config):
    """Entry point for cumulative violations plot."""
    os.makedirs(config.evaluation_dir, exist_ok=True)

    # Load results for all environments in parallel (I/O-bound)
    print("Loading results for all environments in parallel...")
    with ThreadPoolExecutor() as executor:
        futures = {env: executor.submit(load_results_for_env, config, env) for env in config.env_names}
        env_runs = {env: future.result() for env, future in futures.items()}

    # Aggregate stats
    env_stats: dict[str, dict[str, dict[str, Any]]] = {}

    for env_name, runs in env_runs.items():
        if not runs:
            print(f"  No matching results found for {env_name}")
            continue

        runs = trim_runs(runs, config.max_rounds)
        violation_stats = aggregate_cumulative_violations(runs, config.include_uncertain, env_name)
        env_stats[env_name] = violation_stats

        print(f"  {env_name}: {len(runs)} runs")
        for model_name, stats in sorted(violation_stats.items()):
            final_val = stats["mean"][-1] if len(stats["mean"]) > 0 else 0
            print(f"    {model_name}: final fraction = {final_val:.2f}")

    # Check if we have any data
    if not env_stats:
        print("\nNo data found for any environment.")
        return

    # Plot
    fig = plot_cumulative_violations_multi_env(env_stats, config)

    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    if config.format == "pdf":
        fig.savefig(config.save_path, bbox_inches="tight")
    else:
        fig.savefig(config.save_path, dpi=150, bbox_inches="tight")

    print(f"\nSaved plot to {config.save_path}")

    plt.close(fig)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
